#include <iostream>
#include <fstream>
#include <boost/optional.hpp>

#define BOOST_SPIRIT_INCLUDE_PHOENIX
#include <boost/phoenix/phoenix.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include <bayesian/graph.hpp>
#include <bayesian/serializer/bif.hpp>
#include <bayesian/serializer/csv.hpp>
#include <bayesian/serializer/dot.hpp>

struct command_line_t {
    std::string const output;
    std::string const network;
    boost::optional<std::string> const link_info;
};

command_line_t process_command_line(int argc, char* argv[])
{
    boost::program_options::options_description opt("Option");
    opt.add_options()
        ("help,h",                                                  "Show this help")
        ("output,o",  boost::program_options::value<std::string>(), "Output File(.dot)     [required]")
        ("network,n", boost::program_options::value<std::string>(), "Network Input File    [required]")
        ("link,l",    boost::program_options::value<std::string>(), "Link Information File [optional]");

	boost::program_options::variables_map vm;
	store(parse_command_line(argc, argv, opt), vm);
	notify(vm);

    if(vm.count("help"))
    {
        std::cout << opt << std::endl;
        std::exit(0);
    }

    if(!vm.count("output") || !vm.count("network"))
    {
        std::cout << "Required: --output and --network" << std::endl;
        std::cout << opt << std::endl;
        std::exit(0);
    }
    
    if(vm.count("link"))
        return { vm["output"].as<std::string>(), vm["network"].as<std::string>(), vm["link"].as<std::string>()};
    else
        return { vm["output"].as<std::string>(), vm["network"].as<std::string>(), boost::none };
}

int main(int argc, char* argv[])
{
    // コマンドラインパース
    auto const command_line = process_command_line(argc, argv);

    // グラフファイルを開いてgraph_dataに導入
    std::ifstream ifs_graph(command_line.network);
    std::string const graph_data{std::istreambuf_iterator<char>(ifs_graph), std::istreambuf_iterator<char>()};
    ifs_graph.close();
    std::cout << "Loaded Graph: Length = " << graph_data.size() << std::endl;

    // graph_dataよりグラフパース
	bn::graph_t graph;
    bn::database_t data;
    std::tie(graph, data) = bn::serializer::bif().parse(graph_data.cbegin(), graph_data.cend());
    std::cout << "Parsed Graph: Num of Node = " << graph.vertex_list().size() << std::endl;

    // リンクファイルがあるなら，そのリンク状態にする
    if(command_line.link_info)
    {
        // 全てのリンクを削除
        graph.erase_all_edge();

        // リンクファイル読み込み
        std::ifstream ifs(command_line.link_info.get());
        bn::serializer::csv().load(ifs, graph);
        ifs.close();
    }

    // 書出
    std::ofstream ofs(command_line.output);
    bn::serializer::dot().write(ofs, graph, data);
    ofs.close();
}
