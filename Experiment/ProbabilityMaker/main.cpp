// まじめに書かないです

#include <iostream>
#include <functional>
#include <random>

#define BOOST_SPIRIT_INCLUDE_PHOENIX
#include <boost/phoenix/phoenix.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include <bayesian/graph.hpp>
#include <bayesian/utility.hpp>

std::pair<std::string, std::string> process_command_line(int argc, char* argv[])
{
    boost::program_options::options_description opt("Option");
    opt.add_options()
        ("help,h",                                                  "Show this help")
        ("network,n", boost::program_options::value<std::string>(), "Network Path")
        ("sample,s",  boost::program_options::value<std::string>(), "Sample Path");

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
        std::cout << "Need --network and --sample" << std::endl;
        std::cout << opt << std::endl;
        std::exit(0);
    }

    return std::make_pair(vm["network"].as<std::string>(), vm["sample"].as<std::string>());
}

std::vector<double> probability_table(std::size_t const node_selectable, std::mt19937& engine)
{
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double total = 0.00;

    std::vector<double> table;
    for(std::size_t i = 0; i < node_selectable; ++i)
    {
        double const prob = dist(engine);

        table.push_back(prob);
        total += prob;
    }

    std::for_each(
        table.begin(), table.end(),
        [&total](double& value)
        {
            value /= total;
        });

    return table;
}

int main()
{
    //// コマンドラインパース
    //std::string graph_path;
    //std::string sample_path;
    //std::tie(graph_path, sample_path) = process_command_line(argc, argv);

    //// グラフファイルを開いてgraph_dataに導入
    //std::ifstream ifs(graph_path);
    //std::string const graph_data{std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>()};
    //ifs.close();
    //std::cout << "Loaded Graph: Length = " << graph_data.size() << std::endl;

    //// graph_dataよりグラフパース
    //auto const graph = bn::serializer::bif().parse(graph_data.cbegin(), graph_data.cend());
    //auto const& vertex_list = graph.vertex_list();
    //for(std::size_t i = 0; i < vertex_list.size(); ++i)
    //    vertex_list[i]->id = i;
    //std::cout << "Parsed Graph: Num of Node = " << vertex_list.size() << std::endl;

    auto engine = bn::make_engine<std::mt19937>();

    std::size_t node_selectable = 3;
    std::vector<std::vector<std::string>> condition_nodes = {
        {"T", "F", "U"},
        {"T", "F", "U"},
        {"T", "F", "U"},
        {"T", "F", "U"},
        {"T", "F", "U"},
        {"T", "F", "U"},
        {"T", "F", "U"},
        {"T", "F", "U"},
        {"T", "F", "U"}
    };

    std::vector<std::string> selecting(condition_nodes.size());

    std::function<void(std::size_t const)> recursive;
    recursive = [&recursive, &selecting, &condition_nodes, &node_selectable, &engine](std::size_t const index)
    {
        if(index == condition_nodes.size())
        {
            std::cout << "  (" << selecting[0];
            for(std::size_t i = 1; i < selecting.size(); ++i)
                std::cout << ", " << selecting[i];
            std::cout << ") ";

            auto const probability = probability_table(node_selectable, engine);
            std::cout << probability[0];
            for(std::size_t i = 1; i < probability.size(); ++i)
                std::cout << ", " << probability[i];
            std::cout << ";\n";
            return;
        }

        for(auto const& select : condition_nodes[index])
        {
            selecting[index] = select;
            recursive(index + 1);
        }
    };

    recursive(0);
}
