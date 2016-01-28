#include <iostream>
#include <mutex>
#include <vector>
#include <boost/foreach.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/thread.hpp>

#define BOOST_SPIRIT_INCLUDE_PHOENIX
#include <boost/phoenix/phoenix.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

struct commandline_t {
    std::vector<boost::filesystem::path> parent_directories;
    std::size_t eqlist_size;
    std::size_t thread_num;
};

commandline_t process_command_line(int argc, char* argv[])
{
    boost::program_options::options_description opt("Option");
    opt.add_options()
        ("help,h"     ,                                                            "Show this help")
        ("directory,d", boost::program_options::value<std::vector<std::string>>(), "Target Graph Directories")
        ("num,n"      , boost::program_options::value<std::size_t>()             , "E/Q List Size")
        ("thread,j"   , boost::program_options::value<std::size_t>()             , "The number of thread");

    boost::program_options::variables_map vm;
    store(parse_command_line(argc, argv, opt), vm);
    notify(vm);

    if(vm.count("help"))
    {
        std::cout << opt << std::endl;
        std::exit(0);
    }

    if(!vm.count("directory") || !vm.count("num") || !vm.count("thread"))
    {
        std::cout << "Need --directory, --num and --thread" << std::endl;
        std::cout << opt << std::endl;
        std::exit(0);
    }

    std::vector<boost::filesystem::path> paths;
    for(auto const& str : vm["directory"].as<std::vector<std::string>>())
    {
        paths.push_back(str);
    }

    return { std::move(paths), vm["num"].as<std::size_t>(), vm["thread"].as<std::size_t>() };
}

int main(int argc, char* argv[])
{
    // コマンドラインパース
    auto const command_line = process_command_line(argc, argv);

    // 各親ディレクトリの子をコマンドに変換
    std::deque<std::string> commands;
    for(auto const& parent_dir : command_line.parent_directories)
    {
        auto const solver_path = parent_dir / "MAECalculator.exe";
        auto const eqlist_path = parent_dir / "eqlist.csv";

        boost::filesystem::path graph_path, sample_path;
        BOOST_FOREACH(
            boost::filesystem::path const& file_path,
            std::make_pair(boost::filesystem::directory_iterator(parent_dir), boost::filesystem::directory_iterator()))
        {
            if(file_path.extension() == ".bif") graph_path = file_path;
            if(file_path.extension() == ".sample") sample_path = file_path;
        }

        BOOST_FOREACH(
            boost::filesystem::path const& child_dir,
            std::make_pair(boost::filesystem::directory_iterator(parent_dir), boost::filesystem::directory_iterator()))
        {
            if(boost::filesystem::is_directory(child_dir))
            {
                if(boost::filesystem::exists(child_dir / "result.csv"))
                {
                    std::string str;
                    str += "" + boost::filesystem::complete(solver_path).string() + " ";
                    str += "-l \"" + boost::filesystem::complete(eqlist_path).string() + "\" ";
                    str += "-n \"" + boost::filesystem::complete(graph_path).string() + "\" ";
                    str += "-s \"" + boost::filesystem::complete(sample_path).string() + "\" ";
                    str += "-t \"" + boost::filesystem::complete(child_dir).string() + "\" ";
                    str += "-i " + std::to_string(command_line.eqlist_size);

                    commands.push_back(str);
                }
            }
        }
    }

    // commandsのmutex
    std::mutex mutex_;

    // commandをthread_numずつ起動
    boost::thread_group thread_group_;
    for(std::size_t i = 0; i < command_line.thread_num; ++i)
    {
        thread_group_.create_thread(
            [i, &mutex_, &commands]()
            {
                while(true)
                {
                    std::string com;
                    {
                        std::lock_guard<std::mutex> lock_(mutex_);

                        if(commands.empty())
                            return;
                        else
                        {
                            com = commands.back();
                            commands.pop_back();
                        }
                    }

                    std::cout << (std::to_string(i) + ": " + com + "\n");
                    std::system(("start /wait " + com).c_str());
                }
            });
    }

    thread_group_.join_all();
}
