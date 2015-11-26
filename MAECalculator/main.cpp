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

std::size_t const MAE_REPEAT_NUM = 10;
std::size_t const INFERENCE_SAMPLE_SIZE = 1000000;

auto process_command_line(int argc, char* argv[])
    -> std::tuple<std::vector<std::string>, std::string>
{
    boost::program_options::options_description opt("Option");
    opt.add_options()
        ("help,h",                                                                 "Show this help")
        ("directory,d", boost::program_options::value<std::vector<std::string>>(), "Target Graph Directories")
        ("eqlist,l"   , boost::program_options::value<std::string>()             , "Evidence/Query Data Path");

	boost::program_options::variables_map vm;
	store(parse_command_line(argc, argv, opt), vm);
	notify(vm);

    if(vm.count("help"))
    {
        std::cout << opt << std::endl;
        std::exit(0);
    }

    if(!vm.count("directory") || !vm.count("eqlist"))
    {
        std::cout << "Need --directory and --eqlist" << std::endl;
        std::cout << opt << std::endl;
        std::exit(0);
    }

    return std::make_tuple(
        vm["directory"].as<std::vector<std::string>>(),
        vm["eqlist"].as<std::string>()
        );
}

struct calculate_target {
    std::pair<bn::vertex_type, std::size_t> query;
    std::unordered_map<bn::vertex_type, std::size_t> evidence;
    double inference;

    template<class OutputStream>
    OutputStream& write_calculate_target(OutputStream& ost) const
    {
        ost << inference << "\n";

        ost << query.first->id << "," << query.second << "\n";
        ost << evidence.size() << "\n";
        for(auto const& e : evidence)
        {
            ost << e.first->id << "," << e.second << "\n";
        }

        return ost;
    }

    template<class InputStream>
    InputStream& load_calculate_target(InputStream& ist, bn::graph_t const& graph)
    {
        auto const& nodes = graph.vertex_list();

        std::string line;
        
        // 1行目
        std::getline(ist, line);
        inference = std::strtod(line.c_str(), NULL);

        // 2行目
        std::getline(ist, line);
        std::vector<std::string> query_parsed;
        boost::algorithm::split(query_parsed, line, boost::is_any_of(","));
        query = std::make_pair(nodes[std::stoi(query_parsed[0])], std::stoi(query_parsed[1]));

        // 3行目
        std::getline(ist, line);
        auto const evidence_num = std::stoi(line);

        // 4行目~
        for(int i = 0; i < evidence_num; ++i)
        {
            std::getline(ist, line);
            std::vector<std::string> evidence_parsed;
            boost::algorithm::split(evidence_parsed, line, boost::is_any_of(","));
            evidence.insert(std::make_pair(nodes[std::stoi(evidence_parsed[0])], std::stoi(evidence_parsed[1])));
        }

        return ist;
    }
};

std::pair<bn::vertex_type, std::size_t> random_query(std::vector<bn::vertex_type> const& nodes, std::mt19937& engine)
{
    std::uniform_int_distribution<std::size_t> node_dist(0, nodes.size() - 1);
    auto const query_node = nodes[node_dist(engine)];

    std::uniform_int_distribution<std::size_t> select_dist(0, query_node->selectable_num - 1);
    auto const query_value = select_dist(engine);

    return std::make_pair(query_node, query_value);
}

template<class Engine>
std::vector<calculate_target> generate_inference_target(Engine& engine, bn::graph_t const& teacher_graph)
{
    auto const& nodes = teacher_graph.vertex_list();

    // 返すコンテナ
    std::vector<calculate_target> inference_target;
    inference_target.reserve(MAE_REPEAT_NUM);

    // MAE_REPEAT_NUM分のデータを作る
    for(std::size_t i = 0; i < MAE_REPEAT_NUM; ++i)
    {
        // Evidence Nodeの最大数を決定
        std::uniform_int_distribution<std::size_t> node_num_dist(1, teacher_graph.vertex_list().size() / 2);
        std::size_t const maximum_evidence_size = node_num_dist(engine);

        // Evidence Nodeの決定
        std::unordered_map<bn::vertex_type, std::size_t> evidences;
        for(std::size_t i = 0; i < maximum_evidence_size; ++i) // vectorならstd::generateなのに
            evidences.insert(random_query(nodes, engine));

        // Query Nodeの決定
        std::pair<bn::vertex_type, std::size_t> query;
        do query = random_query(nodes, engine);
        while(evidences.find(query.first) != evidences.cend()); // Evidence NodeでないものをQuery Nodeとする

        // 追加
        calculate_target target;
        target.query = std::move(query);
        target.evidence = std::move(evidences);
        inference_target.push_back(
            std::move(target)
            );
    }

    return inference_target;
}

// Mean Absolute Error
double caluculate_mae(bn::graph_t const& graph, bn::sampler const& sampler, std::vector<calculate_target> const target)
{
    // CPTの計算
    sampler.make_cpt(graph);

    // 確率推論器の作成
    bn::inference::likelihood_weighting lhw(graph);

    double mae = 0.0;
    for(auto const& elem : target)
    {
        // 推論
        auto const inference = lhw(elem.evidence, INFERENCE_SAMPLE_SIZE);

        // 差の計算
        mae += std::abs(inference.at(elem.query.first)[0][elem.query.second] - elem.inference) / MAE_REPEAT_NUM;
    }

    return mae;
}

void process_each_graph(bn::graph_t const& teacher_graph, bn::sampler const& sampler, boost::filesystem::path const& result_path, std::vector<calculate_target> const target)
{
    // 作業パス
    boost::filesystem::path const working_directory = result_path.parent_path();

    // result.csvを開き，解析，かつMAE計算
    boost::filesystem::ifstream res_ifs(result_path);
    std::stringstream str;
    std::string tmp;
    int counter = 0;
    double total_mae = 0.0;

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
            line[3] = std::to_string(total_mae / counter);
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

            // MAE計算
            auto const mae = caluculate_mae(graph, sampler, target);
            total_mae += mae;
            line[3] = std::to_string(mae);
            std::cout << mae << std::endl; // Debug

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
        std::tie(target_directories, eqlist_path) = process_command_line(argc, argv);
        std::transform(
            std::begin(target_directories), std::end(target_directories),
            std::back_inserter(target_directory_paths),
            [](std::string const& path) { return boost::filesystem::path(path); }
            );
    }

    // target_directory_pathsの各要素に対して計算を行っていく
    for(auto const& target_directory : target_directory_paths)
    {
        boost::filesystem::path network_path;
        boost::filesystem::path sample_path;
        std::vector<boost::filesystem::path> result_paths;

        BOOST_FOREACH(
            boost::filesystem::path const& path,
            std::make_pair(boost::filesystem::recursive_directory_iterator(target_directory), boost::filesystem::recursive_directory_iterator()))
        {
            if(path.extension() == ".bif")
            {
                network_path = path;
            }
            else if(path.extension() == ".sample")
            {
                sample_path = path;
            }
            else if(path.filename() == "result.csv")
            {
                result_paths.push_back(path);
            }
        }

        std::cout << "Starting..." << std::endl;
        std::cout << "Net: " << network_path << std::endl;
        std::cout << "Sam: " << sample_path << std::endl;
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

        // サンプラに読み込ませる
        bn::sampler sampler;
        sampler.set_filename(sample_path.string());
        sampler.load_sample(teacher_graph.vertex_list());
        std::cout << "Loaded Sample" << std::endl;

        // Evidence/Queryがあれば読み込み，なければ生成
        std::vector<calculate_target> targets;
        if(boost::filesystem::exists(eqlist_path))
        {
            // 読込
            boost::filesystem::ifstream ifs(eqlist_path);
            for(int i = 0; i < MAE_REPEAT_NUM; ++i)
            {
                calculate_target t;
                t.load_calculate_target(ifs, teacher_graph);
                targets.push_back(t);
            }
            ifs.close();
        }
        else
        {
            // 生成
            targets = generate_inference_target<std::mt19937>(engine, teacher_graph);

            // 推論
            sampler.make_cpt(teacher_graph);
            bn::inference::likelihood_weighting lhw(teacher_graph);
            for(auto& target : targets)
            {
                auto const inference = lhw(target.evidence, INFERENCE_SAMPLE_SIZE);
                target.inference = inference.at(target.query.first)[0][target.query.second];
                std::cout << "target.inference = " << target.inference << "\n" << std::endl;
            }

            // 書込
            boost::filesystem::ofstream ofs(eqlist_path);
            for(auto const& eq : targets) eq.write_calculate_target(ofs);
            ofs.close();
        }

        for(auto const& result_path : result_paths)
        {
            std::cout << "Start: " << result_path << std::endl;
            process_each_graph(teacher_graph, sampler, result_path, targets);
        }

        std::cout << std::endl;
    }
}
