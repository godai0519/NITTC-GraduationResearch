#ifndef PRE_EXP_IO_HPP
#define PRE_EXP_IO_HPP

#include <string>
#include <tuple>
#include <vector>
#include <boost/timer/timer.hpp>
#include <boost/filesystem/path.hpp>
#include <bayesian/graph.hpp>
#include "algorithms.hpp"

std::tuple<std::string, std::string, std::string, std::string> process_command_line(int argc, char* argv[]);

std::tuple<bn::graph_t, bn::database_t> load_auto_graph(boost::filesystem::path const& file);

void clear_directory(boost::filesystem::path const& path);

template<class InputStream>
std::vector<std::tuple<bn::vertex_type, bn::vertex_type, double>> mi_list_load(
    InputStream& ist,
    std::vector<bn::vertex_type> const& nodes,
    bn::database_t const& data
    )
{
    auto const search_index =
        [](std::unordered_map<std::size_t, std::string> const& list, std::string const& elem) -> std::size_t
        {
            for(auto it = list.begin(); it != list.end(); ++it)
            {
                if(it->second == elem) return it->first;
            }

            return -1;
        };

    std::vector<std::tuple<bn::vertex_type, bn::vertex_type, double>> res;

    std::string str;
    std::getline(ist, str); std::cout << str << std::endl;
    std::getline(ist, str); std::cout << str << std::endl;
    std::getline(ist, str); std::cout << str << std::endl;

    while(std::getline(ist, str))
    {
        std::vector<std::string> splited_strs;
        boost::algorithm::split(splited_strs, str, boost::is_any_of(","));

        auto const index0 = search_index(data.node_name, splited_strs[0]);
        auto const index1 = search_index(data.node_name, splited_strs[2]);
        auto lhs = nodes[std::min(index0, index1)];
        auto rhs = nodes[std::max(index0, index1)];

        res.emplace_back(lhs, rhs, std::stod(splited_strs[3]));
    }

    return res;
}

template<class OutputStream>
void write_result(
    OutputStream& ost,
    std::vector<result_t> const& all_result
    )
{
    ost << ",Score,Time [s],MAE ,Disappeared Link,Appeared Link,Reversed Link,Change Mutual Information\n";

    double score = 0.0;
    double time = 0.0;
    double mae = 0.0;
    double disappeared_link = 0.0;
    double appeared_link = 0.0;
    double reversed_link = 0.0;
    double change_mi = 0.0;

    auto const all_data_size = all_result.size();
    for(std::size_t i = 0; i < all_data_size; ++i)
    {
        auto const& result = all_result[i];
        ost << i                       << "," ;
        ost << result.score            << "," ;
        ost << result.time             << "," ;
        ost << result.mae              << "," ;
        ost << result.disappeared_link << "," ;
        ost << result.appeared_link    << "," ;
        ost << result.reversed_link    << "," ;
        ost << result.change_mi        << "\n";

        score            += static_cast<double>(result.score)            / all_data_size;
        time             += static_cast<double>(result.time)             / all_data_size;
        mae              += static_cast<double>(result.mae)              / all_data_size;
        disappeared_link += static_cast<double>(result.disappeared_link) / all_data_size;
        appeared_link    += static_cast<double>(result.appeared_link)    / all_data_size;
        reversed_link    += static_cast<double>(result.reversed_link)    / all_data_size;
        change_mi        += static_cast<double>(result.change_mi)        / all_data_size;
    }

    ost << "Ave."           << ",";
    ost << score            << "," ;
    ost << time             << "," ;
    ost << mae              << "," ;
    ost << disappeared_link << "," ;
    ost << appeared_link    << "," ;
    ost << reversed_link    << "," ;
    ost << change_mi        << "\n";
}

#endif
