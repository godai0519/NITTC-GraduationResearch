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

//
//
//struct calculate_target {
//    std::pair<bn::vertex_type, std::size_t> query;
//    std::unordered_map<bn::vertex_type, std::size_t> evidence;
//    double inference;
//
//    template<class OutputStream>
//    OutputStream& write_calculate_target(OutputStream& ost) const
//    {
//        ost << inference << "\n";
//
//        ost << query.first->id << "," << query.second << "\n";
//        ost << evidence.size() << "\n";
//        for(auto const& e : evidence)
//        {
//            ost << e.first->id << "," << e.second << "\n";
//        }
//
//        return ost;
//    }
//
//    template<class InputStream>
//    InputStream& load_calculate_target(InputStream& ist, bn::graph_t const& graph)
//    {
//        auto const& nodes = graph.vertex_list();
//
//        std::string line;
//
//        // 1行目
//        std::getline(ist, line);
//        inference = std::strtod(line.c_str(), NULL);
//
//        // 2行目
//        std::getline(ist, line);
//        std::vector<std::string> query_parsed;
//        boost::algorithm::split(query_parsed, line, boost::is_any_of(","));
//        query = std::make_pair(nodes[std::stoi(query_parsed[0])], std::stoi(query_parsed[1]));
//
//        // 3行目
//        std::getline(ist, line);
//        auto const evidence_num = std::stoi(line);
//
//        // 4行目~
//        for(int i = 0; i < evidence_num; ++i)
//        {
//            std::getline(ist, line);
//            std::vector<std::string> evidence_parsed;
//            boost::algorithm::split(evidence_parsed, line, boost::is_any_of(","));
//            evidence.insert(std::make_pair(nodes[std::stoi(evidence_parsed[0])], std::stoi(evidence_parsed[1])));
//        }
//
//        return ist;
//    }
//};
//
//std::pair<bn::vertex_type, std::size_t> random_query(std::vector<bn::vertex_type> const& nodes, std::mt19937& engine)
//{
//    std::uniform_int_distribution<std::size_t> node_dist(0, nodes.size() - 1);
//    auto const query_node = nodes[node_dist(engine)];
//
//    std::uniform_int_distribution<std::size_t> select_dist(0, query_node->selectable_num - 1);
//    auto const query_value = select_dist(engine);
//
//    return std::make_pair(query_node, query_value);
//}
//
//template<class Engine>
//std::vector<calculate_target> generate_inference_target(Engine& engine, bn::graph_t const& teacher_graph, std::size_t const mae_num)
//{
//    auto const& nodes = teacher_graph.vertex_list();
//
//    // 返すコンテナ
//    std::vector<calculate_target> inference_target;
//    inference_target.reserve(mae_num);
//
//    // MAE_REPEAT_NUM分のデータを作る
//    for(std::size_t i = 0; i < mae_num; ++i)
//    {
//        // Evidence Nodeの最大数を決定
//        std::uniform_int_distribution<std::size_t> node_num_dist(1, teacher_graph.vertex_list().size());
//        std::size_t const maximum_evidence_size = node_num_dist(engine);
//
//        // Evidence Nodeの決定
//        std::unordered_map<bn::vertex_type, std::size_t> evidences;
//        for(std::size_t i = 0; i < maximum_evidence_size; ++i) // vectorならstd::generateなのに
//            evidences.insert(random_query(nodes, engine));
//
//        // Query Nodeの決定
//        std::pair<bn::vertex_type, std::size_t> query;
//        do query = random_query(nodes, engine);
//        while(evidences.find(query.first) != evidences.cend()); // Evidence NodeでないものをQuery Nodeとする
//
//        // 追加
//        calculate_target target;
//        target.query = std::move(query);
//        target.evidence = std::move(evidences);
//        inference_target.push_back(
//            std::move(target)
//            );
//    }
//
//    return inference_target;
//}
//
//// Mean Absolute Error
//std::vector<double> caluculate_mae(bn::graph_t const& graph, bn::sampler const& sampler, std::vector<calculate_target> const target)
//{
//    // CPTの計算
//    sampler.make_cpt(graph);
//
//    // 確率推論器の作成
//    bn::inference::likelihood_weighting lhw(graph);
//
//    std::vector<double> errors;
//    for(auto const& elem : target)
//    {
//        // 推論
//        auto const inference = lhw(elem.evidence, INFERENCE_SAMPLE_SIZE, MAE_ACCURACY);
//
//        // 差の計算
//        auto const difference = std::abs(inference.at(elem.query.first)[0][elem.query.second] - elem.inference);
//        std::cout << "\t" << difference << std::endl;
//        errors.push_back(difference);
//    }
//
//    return errors;
//}
//
//std::tuple<double, double> calc_average_variance(std::vector<double> const& data)
//{
//    double square_sum = 0;
//    double sum = 0;
//
//    for(auto it = data.begin(); it != data.end(); ++it)
//    {
//        square_sum += *it * *it;
//        sum += *it;
//    }
//
//    return std::make_tuple(sum / data.size(), (square_sum / data.size()) - std::pow(sum / data.size(), 2));
//}
//
//void process_calulate_mae(bn::graph_t& teacher_graph, bn::sampler& sample, boost::filesystem::path const& eqlist_path, std::size_t const num, boost::filesystem::path const& path)
//{
//    // Read E/Q list
//    std::vector<calculate_target> eqlist(num);
//    {
//        boost::filesystem::ifstream ifs(eqlist_path);
//        for(auto& elem : eqlist)
//            elem.load_calculate_target(ifs, teacher_graph);
//    }
//
//    // 作業パス
//    boost::filesystem::path const working_directory = path;
//    boost::filesystem::path const result_path = path / "result.csv";
//
//    // result.csvを開き，解析，かつMAE計算
//    boost::filesystem::ifstream res_ifs(result_path);
//    std::stringstream str;
//    std::string tmp;
//    int counter = 0;
//    double total_mae = 0.0;
//
//    // 1行目読み飛ばし
//    std::getline(res_ifs, tmp);
//    str << tmp << "\n";
//
//    while(std::getline(res_ifs, tmp)) // 1行ずつ読み込む
//    {
//        // カンマ区切る
//        std::vector<std::string> line;
//        boost::algorithm::split(line, tmp, boost::is_any_of(","));
//
//        if(line[0] == "Ave.") // 最終行
//        {
//            line[3] = std::to_string(total_mae / counter);
//        }
//        else
//        {
//            // グラフをコピー
//            auto graph = teacher_graph;
//            graph.erase_all_edge();
//
//            // Paths
//            auto const graph_path = working_directory / ("graph" + line[0] + ".csv");
//            auto const errors_path = working_directory / ("graph" + line[0] + "_errors.csv");
//
//            if(boost::filesystem::exists(errors_path))
//            {
//                boost::filesystem::ifstream errors_ifs(errors_path);
//
//                std::string first_line;
//                std::getline(errors_ifs, first_line);
//                first_line.erase(first_line.begin(), first_line.begin() + 9);
//                std::cout << first_line << " ";
//
//                auto const mae = std::stod(first_line);
//                total_mae += mae;
//                line[3] = std::to_string(mae);
//                std::cout << mae << std::endl; // Debug
//            }
//            else
//            {
//                // グラフのCSVのpathを決定
//                boost::filesystem::ifstream graph_ifs(graph_path);
//                bn::serializer::csv().load(graph_ifs, graph);
//                graph_ifs.close();
//
//                // MAE計算
//                auto const errors = caluculate_mae(graph, sample, eqlist);
//                auto const error_info = calc_average_variance(errors);
//                auto const mae = std::accumulate(errors.begin(), errors.end(), 0.0) / num;
//
//                // MAE記録CSV
//                boost::filesystem::ofstream errors_ofs(errors_path);
//                errors_ofs << "Average: " << std::get<0>(error_info) << "\n";
//                errors_ofs << "Variance: " << std::get<1>(error_info) << "\n";
//                errors_ofs << "\n";
//                for(auto const error : errors)
//                    errors_ofs << error << "\n";
//                errors_ofs.close();
//
//                // 加算など
//                total_mae += mae;
//                line[3] = std::to_string(mae);
//
//                std::cout << mae << std::endl; // Debug
//            }
//
//            ++counter;
//        }
//
//        str << boost::algorithm::join(line, ",") << "\n";
//    }
//    res_ifs.close();
//
//    // result.csvを書き直す
//    boost::filesystem::ofstream res_ofs(result_path);
//    res_ofs << str.rdbuf();
//    res_ofs.close();
//}
//
//void process_create_list(bn::graph_t& graph, bn::sampler& sample, boost::filesystem::path const& eqlist_path, std::size_t const num)
//{
//    auto const& vertex_list = graph.vertex_list();
//    std::vector<calculate_target> eqlist;
//
//    // 計算
//    while(true)
//    {
//        // 1つ生成
//        auto elem = generate_inference_target(bn::make_engine<std::mt19937>(), graph, 1)[0];
//
//        std::size_t total_counter = 0;
//        std::size_t counter = 0;
//        for(auto const& samp : sample.table())
//        {
//            // Evidence部分が一致するかどうか
//            bool match = true;
//            for(auto const& evidence : elem.evidence/*s*/)
//            {
//                std::size_t const index = std::distance(
//                    vertex_list.begin(),
//                    std::find(vertex_list.begin(), vertex_list.end(), evidence.first)
//                    );
//
//                if(samp.select.at(index) != evidence.second)
//                {
//                    match = false;
//                    break;
//                }
//            }
//
//            // Queryについてカウント
//            if(match)
//            {
//                total_counter += samp.num;
//
//                std::size_t const index = std::distance(
//                    vertex_list.begin(),
//                    std::find(vertex_list.begin(), vertex_list.end(), elem.query.first)
//                    );
//
//                if(samp.select.at(index) == elem.query.second)
//                {
//                    counter += samp.num;
//                }
//            }
//        }
//
//        // 追加
//        if(total_counter != 0)
//        {
//            auto const inf = static_cast<double>(counter) / total_counter;
//            std::cout << inf << std::endl;
//
//            elem.inference = inf;
//            eqlist.push_back(std::move(elem));
//        }
//
//        if(eqlist.size() == num)
//            break;
//    }
//
//    // 書き出し
//    boost::filesystem::ofstream ofs(eqlist_path);
//    for(auto const& elem : eqlist)
//        elem.write_calculate_target(ofs);
//}
//
//int main(int argc, char* argv[])
//{
//    auto const command_line = process_command_line(argc, argv);
//
//    std::cout << "Starting..." << std::endl;
//    std::cout << "Network: " << command_line.network << std::endl;
//    std::cout << "Samples: " << command_line.sample << std::endl;
//
//    // グラフファイルを開いてgraph_dataに導入
//    std::string const graph_data{std::istreambuf_iterator<char>(boost::filesystem::ifstream(command_line.network)), std::istreambuf_iterator<char>()};
//    std::cout << "Loaded Graph: Length = " << graph_data.size() << std::endl;
//
//    // graph_dataよりグラフパース
//    bn::graph_t graph;
//    bn::database_t data;
//    std::tie(graph, data) = bn::serializer::bif().parse(graph_data.cbegin(), graph_data.cend());
//    std::cout << "Parsed Graph: Num of Node = " << graph.vertex_list().size() << std::endl;
//
//    // サンプラに読み込ませる
//    bn::sampler sampler;
//    sampler.set_filename(command_line.sample.string());
//    sampler.load_sample(graph.vertex_list());
//    std::cout << "Loaded Sample" << std::endl;
//
//    if(command_line.tag_type == commandline_t::tag_t::CALCULATE_MAE)
//    {
//        process_calulate_mae(graph, sampler, command_line.eqlist, command_line.mae_num, command_line.target);
//    }
//    else if(command_line.tag_type == commandline_t::tag_t::CREATE_LIST)
//    {
//        process_create_list(graph, sampler, command_line.eqlist, command_line.mae_num);
//    }
//}
