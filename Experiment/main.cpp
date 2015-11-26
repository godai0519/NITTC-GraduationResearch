#include <iostream>
#include <random>

#define BOOST_SPIRIT_INCLUDE_PHOENIX
#include <boost/phoenix/phoenix.hpp>

#include <bayesian/graph.hpp>
#include <bayesian/sampler.hpp>
#include <bayesian/utility.hpp>
#include <bayesian/inference/likelihood_weighting.hpp>
#include <bayesian/serializer/dot.hpp>
#include <bayesian/serializer/csv.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>

#include "io.hpp"
#include "graph_evaluater.hpp"
#include "experiments.hpp"
#include "algorithms.hpp"
#include <bayesian/serializer/bif.hpp>

int main(int argc, char* argv[])
{
    auto engine = bn::make_engine<std::mt19937>();
    boost::filesystem::path network_path, sample_path, milist_path, output_path;
    std::tie(network_path, sample_path, milist_path, output_path) = process_command_line(argc, argv);

    // �O���t�ǂݍ���
    std::cout << "Load Graph..." << std::endl;
    bn::graph_t teacher_graph;
    bn::database_t teacher_database;
    std::tie(teacher_graph, teacher_database) = load_auto_graph(network_path);

    // �T���v���ɓǂݍ��܂���
    // ����ɋ��t�O���t�ɁC��قǏ���������sampling�f�[�^��p����CPT�č쐬
    std::cout << "Load Sample..." << std::endl;
    bn::sampler sampler;
    sampler.set_filename(sample_path.string());
    sampler.load_sample(teacher_graph.vertex_list());
    sampler.make_cpt(teacher_graph);

    // ���ݏ��ʃ��X�g��ǂݍ���
    std::cout << "Load MI List..." << std::endl;
    boost::filesystem::ifstream mi_ifs(milist_path);
    auto const mi_list = mi_list_load(mi_ifs, teacher_graph.vertex_list(), teacher_database);
    mi_ifs.close();

    // Run!
    for(auto const& algorithm : algorithms)
    {
        std::string const algorithm_name = algorithm.name;
        std::cout << "---------- " << algorithm_name << " ----------" << std::endl;

        std::cout << "Learning..." << std::endl;
        std::vector<result_t> all_result;
        for(std::size_t i = 0; i < iteration_num; ++i)
        {
            // �\���w�K
            auto result = learning(teacher_graph, sampler, algorithm.function);
            sampler.make_cpt(result.graph); // CPT�쐬

            // Score
            result.score = EvaluationAlgorithm(sampler)(result.graph);

            // MI Change
            result.change_mi = distance(teacher_graph, result.graph, mi_list);

            std::cout << "Learned: " << result.time << " (s)" << std::endl;
            all_result.push_back(std::move(result));
        }

        std::cout << "Learning is end\n" << std::endl;
        std::cout << "Write Start" << std::endl;

        // ���ʏ��o�t�H���_�̍쐬
        auto const write_path = output_path / algorithm_name;
        clear_directory(write_path);

        // ���ʂ������o���t�@�C��
        boost::filesystem::ofstream ofs_result(write_path / "result.csv");
        write_result(ofs_result, all_result);
        ofs_result.close();

        // Writers
        bn::serializer::csv csv_io{};
        bn::serializer::dot dot_io{};

        // ���f�[�^�̏����o��
        boost::filesystem::ofstream ofs_original_csv(write_path / "original.csv");
        boost::filesystem::ofstream ofs_original_dot(write_path / "original.dot");
        csv_io.write(ofs_original_csv, teacher_graph);
        dot_io.write(ofs_original_dot, teacher_graph, teacher_database);
        ofs_original_csv.close();
        ofs_original_dot.close();

        // �쐬��O���t�f�[�^�̏����o��
        for(std::size_t i = 0; i < all_result.size(); ++i)
        {
            std::string const filename = "graph" + std::to_string(i);
            boost::filesystem::ofstream ofs_csv(write_path / (filename + ".csv"));
            boost::filesystem::ofstream ofs_dot(write_path / (filename + ".dot"));
            csv_io.write(ofs_csv, all_result[i].graph);
            dot_io.write(ofs_dot, all_result[i].graph, teacher_database);
        }
    }
}
