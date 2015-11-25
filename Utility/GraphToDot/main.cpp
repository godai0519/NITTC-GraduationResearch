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

std::string process_command_line(int argc, char* argv[])
{
    boost::program_options::options_description opt("Option");
    opt.add_options()
        ("help,h",                                                  "Show this help")
        ("network,n", boost::program_options::value<std::string>(), "Network Path");

	boost::program_options::variables_map vm;
	store(parse_command_line(argc, argv, opt), vm);
	notify(vm);

    if(vm.count("help"))
    {
        std::cout << opt << std::endl;
        std::exit(0);
    }

    if(!vm.count("network"))
    {
        std::cout << "Need --network" << std::endl;
        std::cout << opt << std::endl;
        std::exit(0);
    }

    return vm["network"].as<std::string>();
}

int main(int argc, char* argv[])
{
    // コマンドラインパース
    auto const graph_path = process_command_line(argc, argv);

    // グラフファイルを開いてgraph_dataに導入
    std::ifstream ifs(graph_path);
    std::string const graph_data{std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>()};
    ifs.close();
    std::cout << "Loaded Graph: Length = " << graph_data.size() << std::endl;

    // graph_dataよりグラフパース
    bn::graph_t graph;
    bn::database_t data;
    std::tie(graph, data) = bn::serializer::bif().parse(graph_data.cbegin(), graph_data.cend());
    auto const& vertex_list = graph.vertex_list();
    std::cout << "Parsed Graph: Num of Node = " << vertex_list.size() << std::endl;

    // dotファイルを書き出す
    std::ofstream ofs(graph_path + ".dot");
    bn::serializer::dot().write(ofs, graph, data);
    ofs.close();
}
