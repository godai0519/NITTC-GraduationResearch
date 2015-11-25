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
#include <bayesian/serializer/bif.hpp>
#include <bayesian/serializer/csv.hpp>

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

std::size_t count_disappeared_link(bn::graph_t const& teacher, bn::graph_t const& target)
{
    auto const& teacher_edges = teacher.edge_list();
    auto const& target_edges = target.edge_list();

    std::size_t counter = 0;
    for(auto const& teacher_edge : teacher_edges)
    {
        bool is_exist = false;
        for(auto const& target_edge : target_edges)
        {
            auto const is_same = 
                teacher.source(teacher_edge) == target.source(target_edge) &&
                teacher.target(teacher_edge) == target.target(target_edge);

            auto const is_reverse =
                teacher.source(teacher_edge) == target.target(target_edge) &&
                teacher.target(teacher_edge) == target.source(target_edge);

            if(is_same || is_reverse)
            {
                is_exist = true;
                break;
            }
        }

        if(!is_exist) ++counter;

        //auto it = std::find_if(
        //    target_edges.begin(), target_edges.end(),
        //    [&teacher_edge, &teacher, &target](bn::edge_type const& target_edge)
        //    {
        //        return teacher.source(teacher_edge) == target.source(target_edge)
        //            && teacher.target(teacher_edge) == target.target(target_edge);
        //    });
        //
        //if(it == target_edges.end()) ++counter;
    }

    return counter;
}

std::size_t count_appeared_link(bn::graph_t const& teacher, bn::graph_t const& target)
{
    auto const& teacher_edges = teacher.edge_list();
    auto target_edges = target.edge_list();

    for(auto const& teacher_edge : teacher_edges)
    {
        while(true)
        {
            auto const it = std::find_if(
                target_edges.begin(), target_edges.end(),
                [&teacher_edge, &teacher, &target](bn::edge_type const& target_edge)
                {
                    auto const is_same =
                        teacher.source(teacher_edge) == target.source(target_edge) && 
                        teacher.target(teacher_edge) == target.target(target_edge);

                    auto const is_reverse =
                        teacher.source(teacher_edge) == target.target(target_edge) &&
                        teacher.target(teacher_edge) == target.source(target_edge);

                    return is_same || is_reverse;
                });
            
            if(it != target_edges.end()) target_edges.erase(it);
            else                         break;
        }
    }

    return target_edges.size();
}

std::size_t count_reversed_link(bn::graph_t const& teacher, bn::graph_t const& target)
{
    auto const& teacher_edges = teacher.edge_list();
    auto const& target_edges = target.edge_list();

    std::size_t counter = 0;
    for(auto const& teacher_edge : teacher_edges)
    {
        auto it = std::find_if(
            target_edges.begin(), target_edges.end(),
            [&teacher_edge, &teacher, &target](bn::edge_type const& target_edge)
            {
                auto const is_reverse =
                    teacher.source(teacher_edge) == target.target(target_edge) &&
                    teacher.target(teacher_edge) == target.source(target_edge);

                return is_reverse;
            });

        if(it != target_edges.end()) ++counter;
    }

    return counter;
}

double eval(bn::vertex_type const lhs, bn::vertex_type const rhs, std::vector<std::tuple<bn::vertex_type, bn::vertex_type, double>> const& mi_list)
{
    auto it = std::find_if(
        mi_list.begin(), mi_list.end(),
        [&lhs, &rhs](std::tuple<bn::vertex_type, bn::vertex_type, double> const& edge)
        {
            return (std::get<0>(edge) == lhs && std::get<1>(edge) == rhs)
                || (std::get<0>(edge) == rhs && std::get<1>(edge) == lhs);
        });

    if(it == mi_list.end()) return 0.0;
    else                    return std::get<2>(*it);
}

double eval_disappeared_link(bn::graph_t const& teacher, bn::graph_t const& target, std::vector<std::tuple<bn::vertex_type, bn::vertex_type, double>> const& mi_list)
{
    auto const& teacher_edges = teacher.edge_list();
    auto const& target_edges = target.edge_list();

    double decreases = 0.0;
    for(auto const& teacher_edge : teacher_edges)
    {
        bool is_exist = false;
        for(auto const& target_edge : target_edges)
        {
            auto const is_same = 
                teacher.source(teacher_edge) == target.source(target_edge) &&
                teacher.target(teacher_edge) == target.target(target_edge);

            auto const is_reverse =
                teacher.source(teacher_edge) == target.target(target_edge) &&
                teacher.target(teacher_edge) == target.source(target_edge);

            if(is_same || is_reverse)
            {
                is_exist = true;
                break;
            }
        }

        if(!is_exist)
            decreases += eval(teacher.source(teacher_edge), teacher.target(teacher_edge), mi_list);

        //auto it = std::find_if(
        //    target_edges.begin(), target_edges.end(),
        //    [&teacher_edge, &teacher, &target](bn::edge_type const& target_edge)
        //    {
        //        return teacher.source(teacher_edge) == target.source(target_edge)
        //            && teacher.target(teacher_edge) == target.target(target_edge);
        //    });

        //if(it == target_edges.end())
        //{
        //    decreases += eval(teacher.source(teacher_edge), teacher.target(teacher_edge), mi_list);
        //}
    }

    return -decreases;
}

double eval_count_appeared_link(bn::graph_t const& teacher, bn::graph_t const& target, std::vector<std::tuple<bn::vertex_type, bn::vertex_type, double>> const& mi_list)
{
    auto const& teacher_edges = teacher.edge_list();
    auto target_edges = target.edge_list();

    for(auto const& teacher_edge : teacher_edges)
    {
        while(true)
        {
            auto const it = std::find_if(
                target_edges.begin(), target_edges.end(),
                [&teacher_edge, &teacher, &target](bn::edge_type const& target_edge)
                {
                    auto const is_same =
                        teacher.source(teacher_edge) == target.source(target_edge) && 
                        teacher.target(teacher_edge) == target.target(target_edge);

                    auto const is_reverse =
                        teacher.source(teacher_edge) == target.target(target_edge) &&
                        teacher.target(teacher_edge) == target.source(target_edge);

                    return is_same || is_reverse;
                });
            
            if(it != target_edges.end()) target_edges.erase(it);
            else                         break;
        }
        //auto it = std::find_if(
        //    target_edges.begin(), target_edges.end(),
        //    [&teacher_edge, &teacher, &target](bn::edge_type const& target_edge)
        //    {
        //        return (teacher.source(teacher_edge) == target.source(target_edge)
        //                && teacher.target(teacher_edge) == target.target(target_edge))
        //            || (teacher.source(teacher_edge) == target.target(target_edge)
        //                && teacher.target(teacher_edge) == target.source(target_edge));
        //    });

        //if(it != target_edges.end()) target_edges.erase(it);
    }

    double increases = 0.0;
    for(auto const& edge : target_edges)
    {
        increases += eval(target.source(edge), target.target(edge), mi_list);
    }

    return increases;
}

double distance(bn::graph_t const& teacher, bn::graph_t const& graph, std::vector<std::tuple<bn::vertex_type, bn::vertex_type, double>> const& mi_list)
{
    return eval_count_appeared_link(teacher, graph, mi_list)
        + eval_disappeared_link(teacher, graph, mi_list);
}


auto process_command_line(int argc, char* argv[])
    -> std::vector<std::string>
{
    boost::program_options::options_description opt("Option");
    opt.add_options()
        ("help,h",                                                                 "Show this help")
        ("directory,d", boost::program_options::value<std::vector<std::string>>(), "Target Graph Directories");

	boost::program_options::variables_map vm;
	store(parse_command_line(argc, argv, opt), vm);
	notify(vm);

    if(vm.count("help"))
    {
        std::cout << opt << std::endl;
        std::exit(0);
    }

    if(!vm.count("directory"))
    {
        std::cout << "Need --directory" << std::endl;
        std::cout << opt << std::endl;
        std::exit(0);
    }

    return vm["directory"].as<std::vector<std::string>>();
}

void process_each_graph(bn::graph_t const& teacher_graph, boost::filesystem::path const& result_path, std::vector<std::tuple<bn::vertex_type, bn::vertex_type, double>> const& mi_list)
{
    // 作業パス
    boost::filesystem::path const working_directory = result_path.parent_path();

    // result.csvを開き，解析，かつMAE計算
    boost::filesystem::ifstream res_ifs(result_path);
    std::stringstream str;
    std::string tmp;
    int counter = 0;
    
    double total_disappeared_link = 0.0;
    double total_appeared_link = 0.0;
    double total_reversed_link = 0.0;
    double total_change_mi = 0.0;

    // 1行目読み飛ばし
    std::getline(res_ifs, tmp);
    str << tmp << "\n";

    while(std::getline(res_ifs, tmp)) // 1行ずつ読み込む
    {
        // カンマ区切る
        std::vector<std::string> line;
        boost::algorithm::split(line, tmp, boost::is_any_of(","));

        if(line[0] == "Ave.") // 最終行
        {
            line[4] = std::to_string(total_disappeared_link / counter);
            line[5] = std::to_string(total_appeared_link / counter);
            line[6] = std::to_string(total_reversed_link / counter);
            line[7] = std::to_string(total_change_mi / counter);
        }
        else
        {
            // グラフをコピー
            auto graph = teacher_graph;
            graph.erase_all_edge();

            // グラフのCSVのpathを決定
            auto const graph_path = working_directory / ("graph" + line[0] + ".csv");
            boost::filesystem::ifstream graph_ifs(graph_path);
            bn::serializer::csv().load(graph_ifs, graph);

            auto const disappeared_link = count_disappeared_link(teacher_graph, graph);
            auto const appeared_link = count_appeared_link(teacher_graph, graph);
            auto const reversed_link = count_reversed_link(teacher_graph, graph);
            auto const change_mi = distance(teacher_graph, graph, mi_list);
            
            line[4] = std::to_string(disappeared_link);
            line[5] = std::to_string(appeared_link);
            line[6] = std::to_string(reversed_link);
            line[7] = std::to_string(change_mi);

            total_disappeared_link += disappeared_link;
            total_appeared_link += appeared_link;
            total_reversed_link += reversed_link;
            total_change_mi += change_mi;

            ++counter;
        }

        str << boost::algorithm::join(line, ",") << "\n";
    }
    res_ifs.close();

    // result.csvを書き直す
    boost::filesystem::ofstream res_ofs(result_path);
    res_ofs << str.rdbuf();
    res_ofs.close();
}

int main(int argc, char* argv[])
{
    auto engine = bn::make_engine<std::mt19937>();

    // コマンドラインパース
    boost::filesystem::path eqlist_path;
    std::vector<boost::filesystem::path> target_directory_paths;
    {
        std::vector<std::string> target_directories;
        target_directories = process_command_line(argc, argv);
        std::transform(
            std::begin(target_directories), std::end(target_directories),
            std::back_inserter(target_directory_paths),
            [](std::string const& path) { return boost::filesystem::path(path); }
            );
    }

    // target_directory_pathsの各要素に対して計算を行っていく
    for(auto const& target_directory : target_directory_paths)
    {
        boost::filesystem::path network_path, milist_path;
        std::vector<boost::filesystem::path> result_paths;

        BOOST_FOREACH(
            boost::filesystem::path const& path,
            std::make_pair(boost::filesystem::recursive_directory_iterator(target_directory), boost::filesystem::recursive_directory_iterator()))
        {
            if(path.extension() == ".bif")
            {
                network_path = path;
            }
            else if(path.filename() == "mi_list.csv")
            {
                milist_path = path;
            }
            else if(path.filename() == "result.csv")
            {
                result_paths.push_back(path);
            }
        }

        std::cout << "Starting..." << std::endl;
        std::cout << "Network: " << network_path << std::endl;
        std::cout << "MI List: " << milist_path << std::endl;
        for(auto const& result_path : result_paths)
            std::cout << result_path << std::endl;

        // グラフファイルを開いてgraph_dataに導入
        boost::filesystem::ifstream ifs(network_path);
        std::string const graph_data{std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>()};
        ifs.close();
        std::cout << "Loaded Graph: Length = " << graph_data.size() << std::endl;

        // graph_dataよりグラフパース
        bn::graph_t teacher_graph;
        bn::database_t data;
        std::tie(teacher_graph, data) = bn::serializer::bif().parse(graph_data.cbegin(), graph_data.cend());
        std::cout << "Parsed Graph: Num of Node = " << teacher_graph.vertex_list().size() << std::endl;

        // 相互情報量リストを読み込む
        std::cout << "Load MI List..." << std::endl;
        boost::filesystem::ifstream mi_ifs(milist_path);
        auto const mi_list = mi_list_load(mi_ifs, teacher_graph.vertex_list(), data);
        mi_ifs.close();
        
        for(auto const& result_path : result_paths)
        {
            std::cout << "Start: " << result_path << std::endl;
            process_each_graph(teacher_graph, result_path, mi_list);
        }

        std::cout << std::endl;
    }
}
