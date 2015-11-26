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
#include <bayesian/serializer/csv.hpp>
#include <bayesian/serializer/dot.hpp>
#include <bayesian/evaluation/transinformation.hpp>

auto process_command_line(int argc, char* argv[])
    -> std::tuple<std::string, std::string, std::string, std::string>
{
    boost::program_options::options_description opt("Option");
    opt.add_options()
        ("help,h",                                                   "Show this help")
        ("network,n",  boost::program_options::value<std::string>(), "Network Path")
        ("original,g", boost::program_options::value<std::string>(), "Original Structure Path")
        ("target,t",   boost::program_options::value<std::string>(), "Target Structure Path")
        ("output,o",   boost::program_options::value<std::string>(), "Output Path");

	boost::program_options::variables_map vm;
	store(parse_command_line(argc, argv, opt), vm);
	notify(vm);

    if(vm.count("help"))
    {
        std::cout << opt << std::endl;
        std::exit(0);
    }

    if(!vm.count("network") || !vm.count("original") || !vm.count("target") || !vm.count("output"))
    {
        std::cout << "Need --network, --original, --target and --output" << std::endl;
        std::cout << opt << std::endl;
        std::exit(0);
    }

    return std::make_tuple(
        vm["network"].as<std::string>(),
        vm["original"].as<std::string>(),
        vm["target"].as<std::string>(),
        vm["output"].as<std::string>()
        );
}

std::tuple<bn::graph_t, bn::database_t> read_graph(std::string const& path)
{
    // グラフファイルを開いてgraph_dataに導入
    std::ifstream ifs(path);
    std::string const graph_data{std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>()};
    ifs.close();
    std::cout << "Loaded Graph: Length = " << graph_data.size() << std::endl;

    // graph_dataよりグラフパース
    auto data = bn::serializer::bif().parse(graph_data.cbegin(), graph_data.cend());
    auto const& vertex_list = std::get<0>(data).vertex_list();
    std::cout << "Parsed Graph: Num of Node = " << vertex_list.size() << std::endl;
    
    return data;
}

void read_csv(std::string const& path, bn::graph_t& graph)
{
    std::ifstream ifs(path);
    graph.erase_all_edge();
    bn::serializer::csv().load(ifs, graph);

    return;
}

class dot {
public:
    template<class OutputStream>
    OutputStream& write(OutputStream& ost, bn::graph_t const& orig_graph, bn::graph_t const& targ_graph, bn::database_t const& data)
    {
        // 冒頭
        ost << "digraph " << data.graph_name << "{\n";

        // ノードを書き出す
        for(auto const& node : targ_graph.vertex_list())
            ost << "    " << get_node_identify(node) << " [label=\"" << data.node_name.at(node->id) << "\"];\n";

        // エッジを書き出す
        for(auto const& edge : targ_graph.edge_list())
        {
            auto const source = targ_graph.source(edge);
            auto const target = targ_graph.target(edge);
            
            bool is_exist = false;
            for(auto const& orig_edge : orig_graph.edge_list())
            {
                if(
                    (orig_graph.source(orig_edge) == source && orig_graph.target(orig_edge) == target) ||
                    (orig_graph.source(orig_edge) == target && orig_graph.target(orig_edge) == source)
                    )
                {
                    is_exist = true;
                    break;
                }
            }

            ost << "    " << get_node_identify(source) << " -> " << get_node_identify(target);
            if(!is_exist) ost << " [style = dashed]";    
            ost << ";\n";
        }

        ost << "}";
        return ost;
    }

private:
    inline std::string get_node_identify(bn::vertex_type const& node)
    {
        return "Node" + std::to_string(node->id);
    }
};

int main(int argc, char* argv[])
{
    // コマンドラインパース
    std::string network_path, orig_path, targ_path, output_path;
    std::tie(network_path, orig_path, targ_path, output_path) = process_command_line(argc, argv);

    // Graph読み込み
    bn::graph_t graph;
    bn::database_t database;
    std::tie(graph, database) = read_graph(network_path);
    
    // 2グラフ読み込み
    bn::graph_t orig_graph = graph;
    bn::graph_t targ_graph = graph;
    read_csv(orig_path, orig_graph);
    read_csv(targ_path, targ_graph);

    // dotファイルを書き出す
    std::ofstream ofs(output_path);
    dot().write(ofs, orig_graph, targ_graph, database);
    ofs.close();
}
