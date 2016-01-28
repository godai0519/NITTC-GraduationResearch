#define BOOST_SPIRIT_INCLUDE_PHOENIX
#include <fstream>
#include <iostream>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include <bayesian/graph.hpp>
#include <bayesian/sampler.hpp>
#include <bayesian/utility.hpp>
#include <bayesian/serializer/bif.hpp>

struct command_line_t {
    std::string network_file;
    std::string sample_file;
    std::vector<std::string> structure_files;
};

command_line_t process_command_line(int argc, char* argv[])
{
    boost::program_options::options_description opt("Option");
    opt.add_options()
        ("help,h",                                                                 "Show this help")
        ("network,n", boost::program_options::value<std::string>(),                "Target Network [required]")
        ("sample,s",  boost::program_options::value<std::string>(),                "Target Sample [required]")
        ("structure,t", boost::program_options::value<std::vector<std::string>>(), "Target Structures [optional]");

    boost::program_options::variables_map vm;
    store(parse_command_line(argc, argv, opt), vm);
    notify(vm);

    if(vm.count("help"))
    {
        std::cout << opt << std::endl;
        std::exit(0);
    }

    if(!vm.count("network") || !vm.count("sample"))
    {
        std::cout << "Required: network and sample" << std::endl;
        std::cout << opt << std::endl;
        std::exit(0);
    }

    return { vm["network"].as<std::string>(), vm["sample"].as<std::string>(), vm["structure"].as<std::vector<std::string>>() };
}

std::vector<double> calculate_probability(bn::graph_t const& graph, bn::vertex_type const& node)
{
    auto const parents = graph.in_vertexes(node);

    // 親の確率を求める
    std::vector<std::vector<double>> parent_probability;
    parent_probability.reserve(parents.size());
    for(auto const& p : parents)
        parent_probability.push_back(calculate_probability(graph, p));

    std::vector<double> probability(node->selectable_num, 0.0);
    all_combination_pattern(
        parents,
        [&node, &parents, &parent_probability, &probability](bn::condition_t const& cond)
        {
            for(std::size_t i = 0; i < node->selectable_num; ++i)
            {
                double adding_prob = node->cpt[cond].second[i];
                for(std::size_t j = 0; j < parents.size(); ++j)
                {
                    if(parent_probability[j][cond.at(parents[j])] < 1.0e-5)
                    {
                        adding_prob = 0.0;
                        break;
                    }
                    adding_prob *= parent_probability[j][cond.at(parents[j])];
                }

                probability[i] += adding_prob;
            }
        });

    return probability;
}

double calculate_conditional_entropy(bn::graph_t const& graph, bn::vertex_type const& node)
{
    auto const parents = graph.in_vertexes(node);

    // 親の確率を求める
    std::vector<std::vector<double>> parent_probability;
    parent_probability.reserve(parents.size());
    for(auto const& p : parents)
        parent_probability.push_back(calculate_probability(graph, p));

    double entropy = 0.0;
    bn::all_combination_pattern(
        parents,
        [&node, &parents, &parent_probability, &entropy](bn::condition_t const& cond)
        {
            auto const& target_cpt = node->cpt[cond].second;

            for(std::size_t i = 0; i < node->selectable_num; ++i)
            {
                double joined_prob = target_cpt[i];
                for(std::size_t j = 0; j < parents.size(); ++j)
                    joined_prob *= parent_probability[j][cond.at(parents[j])];

                if(joined_prob < 1.0e-5 && target_cpt[i] < 1.0e-5)
                    entropy += 0.0;
                else
                    entropy -= joined_prob * std::log2(target_cpt[i]);
            }
        });

    return entropy;
}

double calculate_entropy_group(bn::graph_t const& graph, std::vector<bn::vertex_type> const& nodes)
{
    double entropy = 0.0;

    auto remain_nodes = nodes;
    while(!remain_nodes.empty())
    {
        // 取り出す (LIFO)
        auto target = remain_nodes.back();
        remain_nodes.pop_back();

        entropy += calculate_conditional_entropy(graph, target);
    }

    return entropy;
}

auto convert_sample_new_graph(std::unordered_map<bn::condition_t, std::size_t> const& sample, std::vector<bn::vertex_type> const& old_vertex, std::vector<bn::vertex_type> const& new_vertex)
    -> std::unordered_map<bn::condition_t, std::size_t>
{
    std::unordered_map<bn::condition_t, std::size_t> new_sample;

    for(auto const& p : sample)
    {
        bn::condition_t new_cond;
        for(auto& one_cond : p.first)
        {
            auto index = std::distance(old_vertex.begin(), std::find(old_vertex.begin(), old_vertex.end(), one_cond.first));
            new_cond[new_vertex[index]] = one_cond.second;
        }

        new_sample[new_cond] = p.second;
    }

    return new_sample;
}

double calculate_maximum_entropy(bn::graph_t const& target_graph, bn::sampler const& sample, std::vector<bn::vertex_type> const& vertex_list)
{
    auto graph = target_graph.clone();
    auto const& nodes = graph.vertex_list();

    // 最大Entropyとなるのは，エッジがひとつも張られていないとき
    graph.erase_all_edge();

    // エッジ無しの場合のCPTを推論する
    bn::sampler sampling;
    sampling.load_sample(nodes, sample.table());
    sampling.make_cpt(graph);

    return calculate_entropy_group(graph, nodes);
}

double calculate_r(bn::graph_t const& graph, bn::sampler const& sample, std::vector<bn::vertex_type> const& vertex_list)
{
    // 最大Entropyを得る
    auto const maximum_entropy = calculate_maximum_entropy(graph, sample, vertex_list);
    std::cout << "Maximum Entropy: " << maximum_entropy << std::endl;

    // 現在のEntropy
    sample.make_cpt(graph);
    auto const present_entropy = calculate_entropy_group(graph, vertex_list);
    std::cout << "Present Entropy: " << present_entropy << std::endl;

    return present_entropy / maximum_entropy;
}

double calculate_r(bn::graph_t const& graph, bn::sampler const& sample)
{
    return calculate_r(graph, sample, graph.vertex_list());
}

double calc_entropy(bn::graph_t const& graph, bn::sampler const& sample)
{
    double entropy = 0.0;
    for(auto const& table_elem : sample.table())
    {
        auto const probs = static_cast<double>(table_elem.num) / sample.sampling_size();
        entropy -= probs * std::log2(probs);
            std::cout << entropy << std::endl;
    }


    /*all_combination_pattern(
        vertex_list,
        [&entropy, &vertex_list, &sample](bn::condition_t const& cond){
            std::vector<std::size_t> select(cond.size());
            std::transform(vertex_list.begin(), vertex_list.end(), select.begin(),
                [&cond](bn::vertex_type const& vertex){ return cond.at(vertex); }
                );

            std::size_t count = 0;
            for(auto const& table_elem : sample.table())
            {
                if(table_elem.select == select)
                    count += table_elem.num;
            }

            if(count != 0)
            {
                auto const probs = static_cast<double>(count) / sample.sampling_size();
                entropy -= probs * std::log2(probs);
            }
            std::cout << entropy << std::endl;
        });*/

    return entropy;
}

int main(int argc, char* argv[])
{
    // コマンドラインパース
    auto const command_line = process_command_line(argc, argv);

    // グラフ読み込み
    std::string graph_data{std::istreambuf_iterator<char>(std::ifstream(command_line.network_file)), std::istreambuf_iterator<char>()};

    // グラフパース
    bn::graph_t graph;
    bn::database_t database;
    std::tie(graph, database) = bn::serializer::bif().parse(graph_data.begin(), graph_data.end());

    // サンプラ読み込み
    bn::sampler sample(command_line.sample_file);
    sample.load_sample(graph.vertex_list());
    sample.make_cpt(graph);

    std::cout << "R = " << calculate_r(graph, sample) << std::endl;
    std::cout << "AAAA = " << calc_entropy(graph, sample) << std::endl;
}
