#include <iostream>
#include <fstream>
#include <vector>

#define BOOST_SPIRIT_INCLUDE_PHOENIX
#include <boost/phoenix/phoenix.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include <bayesian/graph.hpp>
#include <bayesian/serializer/bif.hpp>
#include <bayesian/serializer/csv.hpp>
#include <bayesian/serializer/dot.hpp>

std::tuple<std::string, std::string, std::string> process_command_line(int argc, char* argv[])
{
    boost::program_options::options_description opt("Option");
    opt.add_options()
        ("help,h",                                                  "Show this help")
        ("network,n", boost::program_options::value<std::string>(), "Network Path")
        ("csv,c",     boost::program_options::value<std::string>(), "CSV Path")
        ("dot,d",     boost::program_options::value<std::string>(), "Dot Path");

	boost::program_options::variables_map vm;
	store(parse_command_line(argc, argv, opt), vm);
	notify(vm);

    if(vm.count("help"))
    {
        std::cout << opt << std::endl;
        std::exit(0);
    }

    if(!vm.count("network") || !vm.count("csv") || !vm.count("dot"))
    {
        std::cout << "Need --network, --csv and --dot" << std::endl;
        std::cout << opt << std::endl;
        std::exit(0);
    }

    return std::make_tuple(vm["network"].as<std::string>(), vm["csv"].as<std::string>(), vm["dot"].as<std::string>());
}

int main(int argc, char* argv[])
{
    // コマンドラインパース
    std::string net_path;
    std::string csv_path;
    std::string dot_path;
    std::tie(net_path, csv_path, dot_path) = process_command_line(argc, argv);

    // グラフファイルを開いてgraph_dataに導入
    std::ifstream ifs_graph(net_path);
    std::string const graph_data{std::istreambuf_iterator<char>(ifs_graph), std::istreambuf_iterator<char>()};
    ifs_graph.close();
    std::cout << "Loaded Graph: Length = " << graph_data.size() << std::endl;

    // graph_dataよりグラフパース
	bn::graph_t graph;
    bn::database_t data;
    std::tie(graph, data) = bn::serializer::bif().parse(graph_data.cbegin(), graph_data.cend());
    std::cout << "Parsed Graph: Num of Node = " << graph.vertex_list().size() << std::endl;

    std::ifstream ifs(csv_path);
    graph.erase_all_edge();
    bn::serializer::csv().load(ifs, graph);
    ifs.close();

    std::ofstream ofs(dot_path);
    bn::serializer::dot().write(ofs, graph, data);
    ofs.close();
}