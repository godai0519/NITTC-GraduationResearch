#include <iostream>
#include <cstdlib>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/foreach.hpp>

#define BOOST_SPIRIT_INCLUDE_PHOENIX
#include <boost/phoenix/phoenix.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include <bayesian/serializer/bif.hpp>
#include <bayesian/serializer/dsc.hpp>
#include "io.hpp"

std::tuple<std::string, std::string, std::string, std::string> process_command_line(int argc, char* argv[])
{
    boost::program_options::options_description opt("Option");
    opt.add_options()
        ("help,h",                                                  "Show this help")
        ("network,n", boost::program_options::value<std::string>(), "Network Path")
        ("sample,s",  boost::program_options::value<std::string>(), "Sample Path")
        ("milist,m",  boost::program_options::value<std::string>(), "MI List Path")
        ("output,o",  boost::program_options::value<std::string>(), "Output Directory");

	boost::program_options::variables_map vm;
	store(parse_command_line(argc, argv, opt), vm);
	notify(vm);

    if(vm.count("help"))
    {
        std::cout << opt << std::endl;
        std::exit(0);
    }

    if(!vm.count("network") || !vm.count("sample") || !vm.count("milist") || !vm.count("output"))
    {
        std::cout << "Need \"network\", \"sample\", \"milist\" and \"output\"" << std::endl;
        std::cout << opt << std::endl;
        std::exit(0);
    }

    return std::make_tuple(vm["network"].as<std::string>(), vm["sample"].as<std::string>(), vm["milist"].as<std::string>(), vm["output"].as<std::string>());
}

std::tuple<bn::graph_t, bn::database_t> load_auto_graph(boost::filesystem::path const& file)
{
    // äJÇ¢ÇƒëSïîì«Ç›çûÇﬁ
    boost::filesystem::ifstream ifs(file);
    std::string graph_data{std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>()};

    if(file.extension() == ".bif")
    {
        bn::serializer::bif bif;
        auto&& graph = bif.parse(graph_data.cbegin(), graph_data.cend());
        return graph;
    }
    else if(file.extension() == ".dsc")
        throw std::runtime_error("error: Deprecation of DSC file (" + file.string() + ")");
    else
        throw std::runtime_error("error: Unsupported graph type");
}

void clear_directory(boost::filesystem::path const& path)
{
    boost::system::error_code ec;
    if(boost::filesystem::exists(path, ec))
        boost::filesystem::remove_all(path, ec);
    boost::filesystem::create_directories(path, ec);
}
