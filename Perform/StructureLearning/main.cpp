//#include <iostream>
//#include <random>
//#include <bayesian/graph.hpp>
//#include <bayesian/sampler.hpp>
//#include <bayesian/utility.hpp>
//#include <bayesian/inference/likelihood_weighting.hpp>
//#include <bayesian/serializer/dot.hpp>
//#include <bayesian/serializer/csv.hpp>
//#include <boost/filesystem.hpp>
//#include <boost/filesystem/fstream.hpp>
//
//#include "io.hpp"
//#include "experiments.hpp"
//#include "algorithms.hpp"

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

std::tuple<std::string, std::string> process_command_line(int argc, char* argv[])
{
    boost::program_options::options_description opt("Option");
    opt.add_options()
        ("help,h",                                                 "Show this help")
        ("input,i",  boost::program_options::value<std::string>(), "Input File Path")
        ("output,o", boost::program_options::value<std::string>(), "Output File Path");

	boost::program_options::variables_map vm;
	store(parse_command_line(argc, argv, opt), vm);
	notify(vm);

    if(vm.count("help"))
    {
        std::cout << opt << std::endl;
        std::exit(0);
    }

    if(!vm.count("input") || !vm.count("output"))
    {
        std::cout << "Need \"--input\" and \"--output\"" << std::endl;
        std::cout << opt << std::endl;
        std::exit(0);
    }

    return std::make_tuple(vm["input"].as<std::string>(), vm["output"].as<std::string>());
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
    std::tie(input_file, output_file) = process_command_line(argc, argv);

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
    std::cout << "the number of samples = " << sampler.sampling_size() << std::endl;

    // 機械
    bn::evaluation::entropy ent_machine;

    // エントロピーリスト
    std::vector<double> entropy_list;
    entropy_list.reserve(node_list.size());
    for(auto it = node_list.begin(); it != node_list.end(); ++it)
    {
        entropy_list.push_back(ent_machine(sampler, *it));
        std::cout << (*it)->id << std::endl;
    }

    // 計算をしまう
    auto const maximum_edge = node_list.size() * (node_list.size() - 1) / 2;
    std::vector<std::tuple<bn::vertex_type, bn::vertex_type, double>> mi_list;
    mi_list.reserve(maximum_edge);

    // 相互情報量を計算
    double average_mi = 0.0;
    double maximum_mi = std::numeric_limits<double>::min();
    for(auto outer_it = node_list.begin(); outer_it != node_list.end(); ++outer_it)
    {
        for(auto inner_it = outer_it + 1; inner_it != node_list.end(); ++inner_it)
        {
            auto const mi = entropy_list[(*outer_it)->id] + entropy_list[(*inner_it)->id] - ent_machine(sampler, {*outer_it, *inner_it});
            mi_list.emplace_back(*outer_it, *inner_it, mi);
            std::cout << mi_list.size() << std::endl;

            average_mi += mi / maximum_edge;
            maximum_mi = std::max(maximum_mi, mi);
        }
    }

    // 書出
    std::ofstream ofs(output_file.string() + ".mi.csv");
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

        ofs << database.node_name[std::get<0>(elem)->id] << ",";
        ofs << (is_connect ? "-" : "")  << ",";
        ofs << database.node_name[std::get<1>(elem)->id] << ",";
        ofs << std::get<2>(elem) << "\n";
    }
    ofs.close();

    //std::cout << "learning..." << sampler.sampling_size() << std::endl;
    //bn::learning::stepwise_structure_hc<bn::evaluation::aic, bn::learning::greedy> sshc(sampler);
    //sshc(graph, 0.00);
    //std::cout << "end" << sampler.sampling_size() << std::endl;

    //std::ofstream csv_ofs(output_file.string() + ".csv");
    //bn::serializer::csv().write(csv_ofs, graph);
    //csv_ofs.close();

    //std::ofstream dot_ofs(output_file.string() + ".dot");
    //bn::serializer::dot().write(dot_ofs, graph, database);
    //dot_ofs.close();
}
