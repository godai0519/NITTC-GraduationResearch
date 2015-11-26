#define BOOST_SPIRIT_INCLUDE_PHOENIX
#include <iostream>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/foreach.hpp>
#include <boost/phoenix/phoenix.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <bayesian/sampler.hpp>
#include <bayesian/serializer/bif.hpp>
#include <bayesian/serializer/csv.hpp>
#include <bayesian/utility.hpp>
#include <bayesian/inference/likelihood_weighting.hpp>

#include "eqlist_manager.hpp"
#include "thread_pool.hpp"

struct command_line_t {
    std::vector<std::string> directories;
    std::size_t inference_num;
    std::size_t inference_unit_size;
    std::size_t thread_num;
};

command_line_t process_command_line(int argc, char* argv[])
{
    boost::program_options::options_description opt("Option");
    opt.add_options()
        ("help,h",                                                                            "Show this help")
        ("directory,d", boost::program_options::value<std::vector<std::string>>(),            "Target Directories           [required]")
        ("num,n",       boost::program_options::value<std::size_t>()->default_value(100),     "Inference pattern Num        [optional]")
        ("unit,u",      boost::program_options::value<std::size_t>()->default_value(1000000), "inference Sampling Unit size [optional]")
        ("jobs,j",      boost::program_options::value<std::size_t>()->default_value(1)      , "Jobs size                    [optional]");

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
        std::cout << "Required: some --directory" << std::endl;
        std::cout << opt << std::endl;
        std::exit(0);
    }

    return { vm["directory"].as<std::vector<std::string>>(), vm["num"].as<std::size_t>(), vm["unit"].as<std::size_t>(), vm["jobs"].as<std::size_t>() };
}

std::pair<bn::vertex_type, std::size_t> random_query(std::vector<bn::vertex_type> const& nodes, std::mt19937& engine)
{
    std::uniform_int_distribution<std::size_t> node_dist(0, nodes.size() - 1);
    auto const query_node = nodes[node_dist(engine)];

    std::uniform_int_distribution<std::size_t> select_dist(0, query_node->selectable_num - 1);
    auto const query_value = select_dist(engine);

    return std::make_pair(query_node, query_value);
}
    
// Mean Absolute Error
double caluculate_mae(bn::graph_t const& graph, bn::sampler const& sampler, gr::eqlist_manager const& eqlist, command_line_t const& command_line)
{
    // CPTの計算
    sampler.make_cpt(graph);

    // 確率推論器の作成
    bn::inference::likelihood_weighting lhw(graph);

    double mae = 0.0;
    for(auto const& elem : eqlist.target())
    {
        // 推論
        auto const inference = lhw(elem.evidence, command_line.inference_unit_size);

        // 差の計算
        mae += std::abs(inference.at(elem.query.first)[0][elem.query.second] - elem.inference) / command_line.inference_num;
    }

    return mae;
}


void process_inference(boost::filesystem::path const result_path, bn::graph_t const teacher_graph, bn::sampler const sampler, gr::eqlist_manager const eqlist, command_line_t const command_line)
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
            bn::serializer::csv().load(boost::filesystem::ifstream(graph_path), graph);

            // MAE計算
            auto const mae = caluculate_mae(graph, sampler, eqlist, command_line);
            total_mae += mae;
            line[3] = std::to_string(mae);
            std::cout << result_path.root_directory() << ": " << mae << std::endl; // Debug

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

void analyze_target_directory(gr::thread_pool& threads, std::mt19937& engine, boost::filesystem::path const directory, command_line_t const command_line)
{
    // ディレクトリ構造を解析し，パスを保存
    boost::filesystem::path network_path;
    boost::filesystem::path sample_path;
    boost::filesystem::path eqlist_path = directory / "eqlist.csv";
    std::vector<boost::filesystem::path> result_paths;

    BOOST_FOREACH(
        boost::filesystem::path const& path,
        std::make_pair(boost::filesystem::recursive_directory_iterator(directory), boost::filesystem::recursive_directory_iterator()))
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
    std::cout << "Network: " << network_path << std::endl;
    std::cout <<  "Sample: " << sample_path << std::endl;
    std::cout << " EQList: " << eqlist_path << std::endl;
    for(auto const& result_path : result_paths)
        std::cout << result_path << std::endl;

    // グラフファイルを開いてgraph_dataに導入
    std::string const graph_data{std::istreambuf_iterator<char>(boost::filesystem::ifstream(network_path)), std::istreambuf_iterator<char>()};
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

    sampler.make_cpt(teacher_graph);
    std::cout << "Generate CPT" << std::endl;

    // eqlistが存在しなければ生成
    if(!boost::filesystem::exists(eqlist_path))
    {
        boost::filesystem::ofstream ofs(eqlist_path);

        auto const& nodes = teacher_graph.vertex_list();        
        for(std::size_t i = 0; i < command_line.inference_num; ++i)
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
            
            // 推論
            bn::inference::likelihood_weighting lhw(teacher_graph);
            auto const inferences = lhw(evidences, command_line.inference_unit_size);
                        
            // 書込
            auto const inference = inferences.at(query.first)[0][query.second];
            std::cout << directory.leaf() << ": inference = " << inference << std::endl;
            gr::calculate_target{query, evidences, inference}.write_calculate_target(ofs);
        }
    }

    // eqlist読込
    gr::eqlist_manager eqlist;
    eqlist.load(eqlist_path, teacher_graph.vertex_list(), command_line.inference_num);

    // 全てpost
    for(auto const& result_path : result_paths)
        threads.post(boost::bind(process_inference, result_path, teacher_graph, sampler, eqlist, command_line));
}

int main(int argc, char* argv[])
{
    auto engine = bn::make_engine<std::mt19937>();

    // コマンドラインパース
    auto const command_line = process_command_line(argc, argv);

    // ジョブの用意
    gr::thread_pool threads(command_line.thread_num);
    
    // 全ディレクトリをジョブへ
    for(auto const& target_directory : command_line.directories)
        threads.post(boost::bind(analyze_target_directory, std::ref(threads), std::ref(engine), target_directory, command_line));
}
