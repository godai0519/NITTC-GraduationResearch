#include <iostream>
#include <random>

#define BOOST_SPIRIT_INCLUDE_PHOENIX
#include <boost/phoenix/phoenix.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include <bayesian/graph.hpp>
#include <bayesian/sampler.hpp>
#include <bayesian/utility.hpp>
#include <bayesian/inference/likelihood_weighting.hpp>
#include <bayesian/serializer/bif.hpp>
#include <bayesian/serializer/dot.hpp>
#include <bayesian/evaluation/transinformation.hpp>

auto process_command_line(int argc, char* argv[])
    -> std::tuple<std::string, std::string, std::string>
{
    boost::program_options::options_description opt("Option");
    opt.add_options()
        ("help,h",                                                  "Show this help")
        ("network,n", boost::program_options::value<std::string>(), "Network Path")
        ("sample,s",  boost::program_options::value<std::string>(), "Sample Path")
        ("output,o",  boost::program_options::value<std::string>(), "Output Path");

	boost::program_options::variables_map vm;
	store(parse_command_line(argc, argv, opt), vm);
	notify(vm);

    if(vm.count("help"))
    {
        std::cout << opt << std::endl;
        std::exit(0);
    }

    if(!vm.count("network") || !vm.count("sample") || !vm.count("output"))
    {
        std::cout << "Need --network, --sample and --output" << std::endl;
        std::cout << opt << std::endl;
        std::exit(0);
    }

    return std::make_tuple(
        vm["network"].as<std::string>(),
        vm["sample"].as<std::string>(),
        vm["output"].as<std::string>()
        );
}

int main(int argc, char* argv[])
{
    // コマンドラインパース
    std::string network_path;
    std::string sample_path;
    std::string output_path;
    std::tie(network_path, sample_path, output_path) = process_command_line(argc, argv);

    // グラフファイルを開いてgraph_dataに導入
    std::ifstream ifs(network_path);
    std::string const graph_data{std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>()};
    ifs.close();
    std::cout << "Loaded Graph: Length = " << graph_data.size() << std::endl;

    // graph_dataよりグラフパース
    bn::graph_t graph;
    bn::database_t data;
    std::tie(graph, data) = bn::serializer::bif().parse(graph_data.cbegin(), graph_data.cend());
    auto const& vertex_list = graph.vertex_list();
    auto const& edge_list = graph.edge_list();
    std::cout << "Parsed Graph: Num of Node = " << vertex_list.size() << std::endl;

    for(auto const& vertex : vertex_list)
    {
        std::cout << vertex->id << ": " << vertex->selectable_num << std::endl;
    }

    // サンプラに読み込ませる
    bn::sampler sampler;
    sampler.set_filename(sample_path);
    sampler.load_sample(graph.vertex_list());
    std::cout << "Loaded Sample: " << sampler.sampling_size() << std::endl;

    // 計算をしまう
    auto const maximum_edge = vertex_list.size() * (vertex_list.size() - 1) / 2;
    std::vector<std::tuple<bn::vertex_type, bn::vertex_type, double>> mi_list;
    mi_list.reserve(maximum_edge);

    // 相互情報量を計算
    double average_mi = 0.0;
    double maximum_mi = std::numeric_limits<double>::min();
    for(std::size_t i = 0; i < vertex_list.size(); ++i)
    {
        for(std::size_t j = i + 1; j < vertex_list.size(); ++j)
        {
            bn::evaluation::mutual_information mi_machine;
            auto const mi = mi_machine(sampler, vertex_list[i], vertex_list[j]);
            mi_list.emplace_back(vertex_list[i], vertex_list[j], mi);

            average_mi += mi / maximum_edge;
            maximum_mi = std::max(maximum_mi, mi);

            std::cout << "(" << i << "," << j << ")\n";
        }
    }
    /*
    for(auto outer_it = vertex_list.begin(); outer_it != vertex_list.end(); ++outer_it)
    {
        for(auto inner_it = outer_it + 1; inner_it != vertex_list.end(); ++inner_it)
        {
            bn::evaluation::mutual_information mi_machine;
            auto const mi = mi_machine(sampler, *outer_it, *inner_it);
            mi_list.emplace_back(*outer_it, *inner_it, mi);

            average_mi += mi / maximum_edge;
            maximum_mi = std::max(maximum_mi, mi);
        }
    }
    */

    // 書出
    std::ofstream ofs(output_path);
    ofs << "Average:," << average_mi << "\n";
    ofs << "Maximum:," << maximum_mi << "\n";
    ofs << "\n";

    for(auto const elem : mi_list)
    {
        auto const is_connect = std::any_of(
            edge_list.begin(), edge_list.end(),
            [&graph, &elem](bn::edge_type const& edge)
            {
                auto const source = graph.source(edge);
                auto const target = graph.target(edge);

                if(std::get<0>(elem) == source && std::get<1>(elem) == target)
                    return true;

                if(std::get<1>(elem) == source && std::get<0>(elem) == target)
                    return true;

                return false;
            });

        ofs << data.node_name[std::get<0>(elem)->id] << ",";
        ofs << (is_connect ? "-" : "")  << ",";
        ofs << data.node_name[std::get<1>(elem)->id] << ",";
        ofs << std::get<2>(elem) << "\n";
    }
    ofs.close();
}
