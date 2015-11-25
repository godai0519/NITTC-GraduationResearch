// StructureLearningに組み込むべき

#include <iostream>
#include <regex>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>

#define BOOST_SPIRIT_INCLUDE_PHOENIX
#include <boost/phoenix/phoenix.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include <bayesian/graph.hpp>
#include <bayesian/sampler.hpp>
#include <bayesian/evaluation/aic.hpp>
#include <bayesian/learning/greedy.hpp>
#include <bayesian/learning/stepwise_structure_hc.hpp>
#include <bayesian/serializer/csv.hpp>
#include <bayesian/serializer/dot.hpp>

std::tuple<std::string, std::string, std::string> process_command_line(int argc, char* argv[])
{
    boost::program_options::options_description opt("Option");
    opt.add_options()
        ("help,h",                                                  "Show this help")
        ("input,i",   boost::program_options::value<std::string>(), "Input File Path")
        ("connect,c", boost::program_options::value<std::string>(), "Connection File Path")
        ("output,o",  boost::program_options::value<std::string>(), "Output File Path");

	boost::program_options::variables_map vm;
	store(parse_command_line(argc, argv, opt), vm);
	notify(vm);

    if(vm.count("help"))
    {
        std::cout << opt << std::endl;
        std::exit(0);
    }

    if(!vm.count("input") || !vm.count("output") || !vm.count("connect"))
    {
        std::cout << "Need \"--input\" and \"--output\"" << std::endl;
        std::cout << opt << std::endl;
        std::exit(0);
    }

    return std::make_tuple(vm["input"].as<std::string>(), vm["output"].as<std::string>(), vm["connect"].as<std::string>());
}

template<class InputStream>
bool read_selectable_nums(InputStream& ist, bn::graph_t& graph)
{
    // read one line
    std::string str;
    std::getline(ist, str);

    // parse by comma
    std::vector<std::string> parsed_elements;
    boost::algorithm::split(parsed_elements, str, boost::is_any_of(","));

    // first column is trash, what is there valid data?
    assert(parsed_elements.size() >= 2);

    // create node
    for(auto it = std::next(parsed_elements.cbegin()); it != parsed_elements.cend(); ++it)
    {
        // oops!
        if(it->empty()) break;

        // set new node
        auto const& new_node = graph.add_vertex();
        new_node->id             = graph.vertex_list().size() - 1;
        new_node->selectable_num = std::stoi(*it);
    }

    return true;
}

template<class InputStream>
bool read_node_names(InputStream& ist, bn::graph_t& graph, bn::database_t& database)
{
    // load reference
    auto const& node_list = graph.vertex_list();

    // read one line
    std::string str;
    std::getline(ist, str);

    // parse by comma
    std::vector<std::string> parsed_elements;
    boost::algorithm::split(parsed_elements, str, boost::is_any_of(","));

    // first column is trash, what is there valid data?
    assert(parsed_elements.size() >= 2);
    assert(parsed_elements.size() <= node_list.size() + 1);

    // update database
    for(std::size_t i = 1; i < parsed_elements.size(); ++i)
    {
        // oops!
        if(parsed_elements[i].empty()) break;

        // set node name
        auto const& node = node_list[i - 1];
        database.node_name[node->id] = parsed_elements[i];
    }

    return true;
}

template<class InputStream>
bool read_node_options(InputStream& ist, bn::graph_t& graph, bn::database_t& database)
{
    // load reference
    auto const& node_list = graph.vertex_list();

    // read one line
    std::string str;
    std::getline(ist, str);

    // parse by comma
    std::regex pattern(R"(,"[^"]*")");
    std::regex_token_iterator<std::string::iterator> first(str.begin(), str.end(), pattern), last;
    std::vector<std::vector<std::string>> options;
    while(first != last)
    {
        auto parsed = first->str();
        parsed.pop_back();
        parsed.pop_back();
        parsed.erase(0, 3);

        std::vector<std::string> option;
        boost::algorithm::split(option, parsed, boost::is_any_of(","));
        options.push_back(option);

        ++first;
    }

    for(std::size_t i = 0; i < node_list.size(); ++i)
    {
        database.options_name[node_list[i]->id] = options[i];
    }

    return true;
}

template<class InputStream>
bn::condition_t read_sample(InputStream& ist, std::vector<bn::vertex_type> const& node_list, bn::database_t& database)
{
    bn::condition_t condition;

    // read one line
    std::string str;
    if(!std::getline(ist, str)) return condition;

    // parse by comma
    std::vector<std::string> parsed_elements;
    boost::algorithm::split(parsed_elements, str, boost::is_any_of(","));

    // first column is trash, what is there valid data?
    assert(parsed_elements.size() >= 2);
    assert(parsed_elements.size() <= node_list.size() + 1);

    // make condition(sample)
    for(std::size_t i = 1; i < parsed_elements.size(); ++i)
    {
        // oops!
        if(parsed_elements[i].empty()) break;

        auto const& node = node_list[i - 1];
        auto const& data = database.options_name[node->id];
        auto const select = std::distance(data.begin(), std::find(data.begin(), data.end(), parsed_elements[i]));
        assert(select >= 0 && select < node->selectable_num);

        condition[node] = select;
    }

    return condition;
}

int main(int argc, char* argv[])
{
    // parse command line argument
    boost::filesystem::path input_file;
    boost::filesystem::path output_file;
    boost::filesystem::path connect_file;
    std::tie(input_file, output_file, connect_file) = process_command_line(argc, argv);

    // initialize graph
    bn::graph_t graph;
    bn::database_t database;
    auto const& node_list = graph.vertex_list();
    auto const& edge_list = graph.edge_list();

    // open input file
    boost::filesystem::ifstream ifs(input_file);

    // first line and second line
    read_selectable_nums(ifs, graph);
    read_node_options(ifs, graph, database);
    read_node_names(ifs, graph, database);
    std::cout << "the number of nodes = " << node_list.size() << std::endl;

    //
    boost::filesystem::ifstream ifs_connect(connect_file);
    bn::serializer::csv().load(ifs_connect, graph);
    ifs_connect.close();
    std::cout << "finished load connection" << std::endl;

    // make samples for bn::sampler
    std::unordered_map<bn::condition_t, std::size_t> samples;
    while(true)
    {
        // sample line
        auto const sample = read_sample(ifs, node_list, database);
        if(sample.size() != node_list.size()) break;

        // add sample or count sample
        auto const it = samples.lower_bound(sample);
        if(it == samples.end() || it->first != sample)
        {
            samples.emplace_hint(
                it,
                std::piecewise_construct,
                std::forward_as_tuple(sample),
                std::forward_as_tuple(1)
                );
        }
        else
        {
            ++(it->second);
        }
    }
    std::cout << "finished reading sample" << std::endl;

    // make sampler by samples
    bn::sampler sampler;
    sampler.load_sample(samples);
    sampler.make_cpt(graph);
    std::cout << "the number of samples = " << sampler.sampling_size() << std::endl;

    // CPT書き出し
    boost::filesystem::ofstream ofs(output_file);
    for(auto const& node : node_list)
    {
        //
        // 1行目
        //
        auto const& condition_node_list = node->cpt.condition_node();
        for(auto const& condition_node : condition_node_list)
            ofs << database.node_name[condition_node->id] << ",";

        for(std::size_t i = 0; i < node->selectable_num; ++i)
        {
            if(i != 0) ofs << ",";
            ofs << database.node_name[node->id] << " = " << database.options_name[node->id][i];
        }

        ofs << "\n";

        //
        // 2行目以降
        //
        auto const function = [&](bn::condition_t const& cond) -> void
        {
            // condition_node_list順に表示
            for(auto const& condition_node : condition_node_list)
                ofs << cond.at(condition_node) << ",";

            // 確率表示
            auto const& probabilities = node->cpt[cond].second;
            for(std::size_t i = 0; i < probabilities.size(); ++i)
            {
                if(i != 0) ofs << ",";
                ofs << probabilities[i];
            }

            // 改行
            ofs << "\n";
        };

        // CPTの数繰り返す
        bn::all_combination_pattern(condition_node_list, function);
        ofs << "\n";

/*
(W | X, Y, Z)のとき

X,Y,Z, W=0, W=1
0,0,0, 1.0, 0.0
0,0,1, 0.3, 0.7
0,1,0, 1.0, 0.0
*/
    }

    ofs.close();
}
