#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>

#define BOOST_SPIRIT_INCLUDE_PHOENIX
#include <boost/phoenix/phoenix.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

struct command_line_t {
    std::string const from;
    std::vector<std::string> const join;
};

struct result_t {
    boost::filesystem::path src_path;
    double      score;
    double      time;
    double      mae;
    std::size_t disappeared_link;
    std::size_t appeared_link;
    std::size_t reversed_link;
    double      change_mi;
};

command_line_t process_command_line(int argc, char* argv[])
{
    boost::program_options::options_description opt("Option");
    opt.add_options()
        ("help,h",                                                            "Show this help")
        ("from,f", boost::program_options::value<std::string>()             , "Source Directory       [required]")
        ("join,j", boost::program_options::value<std::vector<std::string>>(), "Inner-join Directories [required]");

	boost::program_options::variables_map vm;
	store(parse_command_line(argc, argv, opt), vm);
	notify(vm);

    if(vm.count("help"))
    {
        std::cout << opt << std::endl;
        std::exit(0);
    }

    if(!vm.count("from") || !vm.count("join"))
    {
        std::cout << "Required: --from and --join" << std::endl;
        std::cout << opt << std::endl;
        std::exit(0);
    }
 
    return { vm["from"].as<std::string>(), vm["join"].as<std::vector<std::string>>() };
}

std::vector<boost::filesystem::path> search_target_directories(boost::filesystem::path const& directory_path)
{
    std::vector<boost::filesystem::path> result;

    BOOST_FOREACH(
        boost::filesystem::path const& path,
        std::make_pair(boost::filesystem::directory_iterator(directory_path), boost::filesystem::directory_iterator())
        )
    {
        if(boost::filesystem::is_directory(path))
        {
            if(boost::filesystem::exists(path / "result.csv"))
            {
                result.push_back(path);
            }
        }
    }

    return result;
}

std::vector<result_t> analyze_result(boost::filesystem::path const& path)
{
    boost::filesystem::ifstream ifs(path);

    std::string tmp;
    std::getline(ifs, tmp);

    std::vector<result_t> analyzed;
    while(true)
    {
        std::vector<std::string> split_values;
        std::getline(ifs, tmp);
        boost::algorithm::split(split_values, tmp, boost::is_any_of(","));

        if(split_values[0] == "Ave.")
            break;

        result_t res;
        res.src_path         = path.branch_path() / ("graph" + split_values[0] + ".csv");
        res.score            = std::stod(split_values[1]);
        res.time             = std::stod(split_values[2]);
        res.mae              = std::stod(split_values[3]);
        res.disappeared_link = std::stoull(split_values[4]);
        res.appeared_link    = std::stoull(split_values[5]);
        res.reversed_link    = std::stoull(split_values[6]);
        res.change_mi        = std::stod(split_values[7]);

        if(!boost::filesystem::exists(res.src_path))
            throw std::runtime_error("No file: " + res.src_path.string());

        analyzed.emplace_back(std::move(res));
    }

    return analyzed;
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

void joint_result_directories(
    std::string const& setting_name,
    boost::filesystem::path const& from_dir,
    std::vector<boost::filesystem::path> const& join_dirs
    )
{
    auto const result_path = from_dir / setting_name / "result.csv";
    auto target_result = analyze_result(result_path);
    std::vector<std::vector<result_t>> join_target_result;
    for(auto const& dir : join_dirs)
    {
        auto const path = dir / setting_name / "result.csv";
        if(boost::filesystem::exists(path))
            join_target_result.push_back(analyze_result(path));
    }

    std::size_t next_number = target_result.size();
    for(auto const& joining_result : join_target_result)
    {
        for(auto current_joining : joining_result)
        {
            auto const number_str = std::to_string(next_number++);

            // ファイルの移動
            auto const dst_path = from_dir / setting_name;
            boost::filesystem::rename(current_joining.src_path, dst_path / ("graph" + number_str + ".csv"));
            boost::filesystem::rename(current_joining.src_path.replace_extension(".dot"), dst_path / ("graph" + number_str + ".dot"));

            target_result.emplace_back(std::move(current_joining));
        }
    }

    boost::filesystem::ofstream ofs(result_path);
    write_result(ofs, target_result);
}

int main(int argc, char* argv[])
{
    // コマンドラインパース
    auto const command_line = process_command_line(argc, argv);

    std::vector<boost::filesystem::path> join_paths;
    std::transform(
        command_line.join.begin(), command_line.join.end(), std::back_inserter(join_paths),
        [](std::string const& path) -> boost::filesystem::path { return path; }
        );

    for(auto const& target_directory : search_target_directories(command_line.from))
    {
        joint_result_directories(target_directory.stem().string(), command_line.from, join_paths);
    }



 //   // グラフファイルを開いてgraph_dataに導入
 //   std::ifstream ifs_graph(command_line.network);
 //   std::string const graph_data{std::istreambuf_iterator<char>(ifs_graph), std::istreambuf_iterator<char>()};
 //   ifs_graph.close();
 //   std::cout << "Loaded Graph: Length = " << graph_data.size() << std::endl;

 //   // graph_dataよりグラフパース
	//bn::graph_t graph;
 //   bn::database_t data;
 //   std::tie(graph, data) = bn::serializer::bif().parse(graph_data.cbegin(), graph_data.cend());
 //   std::cout << "Parsed Graph: Num of Node = " << graph.vertex_list().size() << std::endl;

 //   // リンクファイルがあるなら，そのリンク状態にする
 //   if(command_line.link_info)
 //   {
 //       // 全てのリンクを削除
 //       graph.erase_all_edge();

 //       // リンクファイル読み込み
 //       std::ifstream ifs(command_line.link_info.get());
 //       bn::serializer::csv().load(ifs, graph);
 //       ifs.close();
 //   }

 //   // 書出
 //   std::ofstream ofs(command_line.output);
 //   bn::serializer::dot().write(ofs, graph, data);
 //   ofs.close();
}
