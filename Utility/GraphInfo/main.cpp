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

std::pair<std::size_t, std::size_t> get_node_value_info(bn::graph_t const& graph)
{
    std::size_t max_valable = 0;
    std::size_t min_valable = std::numeric_limits<std::size_t>::max();

    for(auto const& node : graph.vertex_list())
    {
        max_valable = std::max(max_valable, node->selectable_num);
        min_valable = std::min(min_valable, node->selectable_num);
    }

    return std::make_pair(min_valable, max_valable);
}

double calc_in_degree(bn::graph_t const& graph)
{
    std::size_t sum_degree = 0;
    for(auto const& node : graph.vertex_list())
        sum_degree += graph.in_edges(node).size();

    return static_cast<double>(sum_degree) / graph.vertex_list().size();
}

double calc_out_degree(bn::graph_t const& graph)
{
    std::size_t sum_degree = 0;
    for(auto const& node : graph.vertex_list())
        sum_degree += graph.out_edges(node).size();

    return static_cast<double>(sum_degree) / graph.vertex_list().size();
}

std::size_t calc_freedom(bn::graph_t const& graph)
{
    std::size_t freedom = 1;
    for(auto const& node : graph.vertex_list())
        freedom *= node->selectable_num;

    return freedom;
}

int main(int argc, char* argv[])
{
    // コマンドラインパース
    std::string const net_path = process_command_line(argc, argv);

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

    auto const valable_info = get_node_value_info(graph);
    auto const in_degree = calc_in_degree(graph);
    auto const out_degree = calc_out_degree(graph);
    auto const freedom = calc_freedom(graph);

    std::cout << "Num of node: " << graph.vertex_list().size() << std::endl;
    std::cout << "Num of edge: " << graph.edge_list().size()   << std::endl;
    std::cout << "Value Range: " << valable_info.first << "~" << valable_info.second << std::endl;
    std::cout << "  in-degree: " << in_degree << std::endl;
    std::cout << " out-degree: " << out_degree << std::endl;
    std::cout << "Deg Freedom: " << freedom << std::endl;
}
