#include <iostream>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/foreach.hpp>
#include <boost/algorithm/string/split.hpp>

#define BOOST_SPIRIT_INCLUDE_PHOENIX
#include <boost/phoenix/phoenix.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include <bayesian/graph.hpp>
#include <bayesian/sampler.hpp>
#include <bayesian/utility.hpp>
#include <bayesian/inference/likelihood_weighting.hpp>
#include <bayesian/inference/belief_propagation.hpp>
#include <bayesian/serializer/bif.hpp>
#include <bayesian/serializer/csv.hpp>

struct commandline_t
{
    boost::filesystem::path output;
    std::vector<boost::filesystem::path> directories;
    std::vector<std::size_t> sizes;
};

struct mae_unit
{
    std::size_t size;
    boost::filesystem::path eqlist;
    std::vector<boost::filesystem::path> targets;
};

commandline_t process_command_line(int argc, char* argv[])
{
    boost::program_options::options_description opt("Option");
    opt.add_options()
        ("help,h"     ,                                                            "Show this help")
        ("output,o"   , boost::program_options::value<std::string>()             , "Output Directory")
        ("directory,d", boost::program_options::value<std::vector<std::string>>(), "Merging target Directories")
        ("num,i"      , boost::program_options::value<std::vector<std::size_t>>(), "Merging target Directory sizes");

    boost::program_options::variables_map vm;
    store(parse_command_line(argc, argv, opt), vm);
    notify(vm);

    if(vm.count("help"))
    {
        std::cout << opt << std::endl;
        std::exit(0);
    }

    if(!vm.count("output") || !vm.count("directory") || !vm.count("num"))
    {
        std::cout << "Need --output, --directory and --num" << std::endl;
        std::cout << opt << std::endl;
        std::exit(0);
    }

    auto directories = vm["directory"].as<std::vector<std::string>>();
    auto sizes = vm["num"].as<std::vector<std::size_t>>();
    if(directories.size() != sizes.size())
    {
        std::cout << "the number of --directory and the number of --num are equals." << std::endl;
        std::cout << opt << std::endl;
        std::exit(0);
    }

    std::vector<boost::filesystem::path> directory_paths;
    for(auto const& dir : directories)
        directory_paths.push_back(dir);

    return commandline_t{
        vm["output"].as<std::string>(),
        directory_paths,
        sizes
    };
}

std::vector<mae_unit> lockup_units(
    std::vector<boost::filesystem::path> const& directories,
    std::vector<std::size_t> const sizes
    )
{
    assert(directories.size() == sizes.size());

    std::vector<mae_unit> units;
    for(std::size_t i = 0; i < directories.size(); ++i)
    {
        boost::filesystem::path eqlist;
        std::vector<boost::filesystem::path> targets;
        
        // 必要ファイルのリストアップ
        BOOST_FOREACH(
            boost::filesystem::path const& path,
            std::make_pair(boost::filesystem::directory_iterator(directories[i]), boost::filesystem::directory_iterator())
            )
        {
            if(boost::filesystem::is_directory(path) && boost::filesystem::exists(path / "result.csv"))
            {
                targets.push_back(path);
            }
            else if(path.filename() == "eqlist.csv")
            {
                eqlist = path;
            }
        }

        // Directory名のソート
        std::sort(
            targets.begin(), targets.end(),
            [](boost::filesystem::path const& lhs, boost::filesystem::path const& rhs){ return lhs.stem() < rhs.stem(); }
            );

        units.push_back(mae_unit{sizes[i], eqlist, targets});
    }

    return units;
}

std::tuple<double, double> calc_average_variance(std::vector<double> const& data)
{
    double square_sum = 0;
    double sum = 0;

    for(auto it = data.begin(); it != data.end(); ++it)
    {
        square_sum += *it * *it;
        sum += *it;
    }

    return std::make_tuple(sum / data.size(), (square_sum / data.size()) - std::pow(sum / data.size(), 2));
}

void joint_all_unit(boost::filesystem::path const& output, std::vector<mae_unit> const& units)
{
    assert(boost::filesystem::is_directory(output));
    assert(!units.empty());
    for(std::size_t i = 1; i < units.size(); ++i)
        assert(units[i-1].targets.size() == units[i].targets.size());

    // マージ対象の数．merge_num個のディレクトリに纏められる
    auto const merge_num = units[0].targets.size();

    // 各手法を回す
    for(std::size_t i = 0; i < merge_num; ++i)
    {
        std::vector<double> maes;
        
        // result.csvを開く
        boost::filesystem::ifstream ifs(units[0].targets[i] / "result.csv");
        std::stringstream ss;

        std::string result_tmp;
        std::getline(ifs, result_tmp);
        ss << result_tmp << "\n";

        while(std::getline(ifs, result_tmp))
        {
            std::vector<std::string> lines;
            boost::algorithm::split(lines, result_tmp, boost::is_any_of(","));
            if(lines[0] == "Ave.")
            {
                // 平均行を書き出して終了
                lines[3] = std::to_string(std::accumulate(maes.begin(), maes.end(), 0.0) / maes.size());
                ss << boost::algorithm::join(lines, ",") << "\n";
                break;
            }

            auto const errorlist_name = "graph" + lines[0] + "_errors.csv";
            std::vector<double> errors;

            // 各ユニットを回す
            for(std::size_t j = 0; j < units.size(); ++j)
            {
                boost::filesystem::ifstream error_ifs(units[j].targets[i] / errorlist_name);

                std::string error_tmp;
                std::getline(error_ifs, error_tmp);
                std::getline(error_ifs, error_tmp);
                std::getline(error_ifs, error_tmp);

                for(std::size_t k = 0; k < units[j].size; ++k)
                {
                    std::getline(error_ifs, error_tmp);
                    errors.push_back(std::stod(error_tmp));
                }
            }
            
            // MAE計算
            auto const error_ave_var = calc_average_variance(errors);
            auto const mae = std::get<0>(error_ave_var);
            maes.push_back(mae);

            // result.csvの1行書出し
            lines[3] = std::to_string(mae);
            ss << boost::algorithm::join(lines, ",") << "\n";

            // *_error.csv書出し
            boost::filesystem::ofstream errors_ofs(output / units[0].targets[i].stem() / errorlist_name);
            errors_ofs << "Average: " << std::get<0>(error_ave_var) << "\n";
            errors_ofs << "Variance: " << std::get<1>(error_ave_var) << "\n";
            errors_ofs << "\n";
            for(auto const error : errors)
                errors_ofs << error << "\n";
            errors_ofs.close();
        }
        ifs.close();

        // result.csvの書出し
        boost::filesystem::ofstream ofs(output / units[0].targets[i].stem() / "result.csv");
        ofs << ss.rdbuf();
        ofs.close();
    }

    // eqlist.csv
    boost::filesystem::ofstream ofs(output / "eqlist.csv");
    for(std::size_t i = 0; i < units.size(); ++i)
    {
        boost::filesystem::ifstream ifs(units[i].eqlist);
        ofs << ifs.rdbuf();
        ifs.close();
    }
    ofs.close();
}

int main(int argc, char* argv[])
{
    auto const commandline = process_command_line(argc, argv);
    auto const units = lockup_units(commandline.directories, commandline.sizes);
    joint_all_unit(commandline.output, units);
}
