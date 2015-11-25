#ifndef PRE_EXP_IO_HPP
#define PRE_EXP_IO_HPP

#include <string>
#include <tuple>
#include <vector>
#include <boost/timer/timer.hpp>
#include <boost/filesystem/path.hpp>
#include <bayesian/graph.hpp>
#include "algorithms.hpp"

std::tuple<std::string, std::string, std::string> process_command_line(int argc, char* argv[]);

std::tuple<bn::graph_t, bn::database_t> load_auto_graph(boost::filesystem::path const& file);

void clear_directory(boost::filesystem::path const& path);

template<class OutputStream>
void write_result(
    OutputStream& ost,
    std::vector<result_t> const& all_result
    )
{
    ost << ",graph score,time [ms],MAE [%],disappeared link,appeared_link\n";

    double score = 0.0;
    double time = 0.0;
    double mae = 0.0;
    double disappeared_link = 0.0;
    double appeared_link = 0.0;

    auto const all_data_size = all_result.size();
    for(std::size_t i = 0; i < all_data_size; ++i)
    {
        auto const& result = all_result[i];
        ost << i                       << "," ;
        ost << result.score            << "," ;
        ost << result.time             << "," ;
        ost << result.mae              << "," ;
        ost << result.disappeared_link << "," ;
        ost << result.appeared_link    << "\n";

        score            += static_cast<double>(result.score)            / all_data_size;
        time             += static_cast<double>(result.time)             / all_data_size;
        mae              += static_cast<double>(result.mae)              / all_data_size;
        disappeared_link += static_cast<double>(result.disappeared_link) / all_data_size;
        appeared_link    += static_cast<double>(result.appeared_link)    / all_data_size;
    }

    ost << "Ave."           << ",";
    ost << score            << "," ;
    ost << time             << "," ;
    ost << mae              << "," ;
    ost << disappeared_link << "," ;
    ost << appeared_link    << "\n";
}

#endif
