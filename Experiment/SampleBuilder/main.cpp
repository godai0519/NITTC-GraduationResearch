#include <iostream>
#include <fstream>

#define DEBUG_LOG_ 1
#define BOOST_SPIRIT_INCLUDE_PHOENIX
#include <boost/phoenix/phoenix.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include <bayesian/graph.hpp>
#include <bayesian/serializer/bif.hpp>
#include <bayesian/inference/likelihood_weighting.hpp>

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

template<class OutputStream>
void write_sample(OutputStream& ost, std::vector<bn::vertex_type> const& vertex_list, std::vector<bn::inference::likelihood_weighting::element_type> const& samples)
{
    for(auto const& data : samples)
    {
        // dataの列をつくる
        std::ostringstream oss;
        for(std::size_t i = 0; i < data.select.size(); ++i)
        {
            if(i != 0) oss << " ";
            oss << data.select[i];
        }
        oss << "\n";

        // data_num回として書き出す
        ost << data.num << " " << oss.str();
    }
}

int main(int argc, char* argv[])
{
    // コマンドラインパース
    std::string graph_path;
    std::string sample_path;
    std::tie(graph_path, sample_path) = process_command_line(argc, argv);

    // グラフファイルを開いてgraph_dataに導入
    std::ifstream ifs(graph_path);
    std::string const graph_data{std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>()};
    ifs.close();
    std::cout << "Loaded Graph: Length = " << graph_data.size() << std::endl;

    // graph_dataよりグラフパース
    auto const graph = std::get<0>(bn::serializer::bif().parse(graph_data.cbegin(), graph_data.cend()));
    auto const& vertex_list = graph.vertex_list();
    std::cout << "Parsed Graph: Num of Node = " << vertex_list.size() << std::endl;

    // サンプルのmake
    bn::inference::likelihood_weighting lw(graph);
    auto const samples = lw.make_samples({}, 2000000);

    // サンプルの書出
    std::ofstream ofs(sample_path);
    write_sample(ofs, vertex_list, samples);
    ofs.close();
}
