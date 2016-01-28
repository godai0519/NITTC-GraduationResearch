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
#include <bayesian/evaluation/transinformation.hpp>
#include <bayesian/inference/likelihood_weighting.hpp>
#include <bayesian/inference/belief_propagation.hpp>
#include <bayesian/serializer/bif.hpp>
#include <bayesian/serializer/csv.hpp>

struct commandline_t
{
    boost::filesystem::path network;
    boost::filesystem::path sample;
    boost::filesystem::path target;
    boost::filesystem::path log;
    boost::filesystem::path output;
};

commandline_t process_command_line(int argc, char* argv[])
{
    boost::program_options::options_description opt("Option");
    opt.add_options()
        ("help,h"   ,                                               "Show this help")
        ("network,n", boost::program_options::value<std::string>(), "Network path")
        ("sample,s" , boost::program_options::value<std::string>(), "Sample path")
        ("target,t" , boost::program_options::value<std::string>(), "Target directory")
        ("log,l"    , boost::program_options::value<std::string>(), "Load log file path")
        ("output,o" , boost::program_options::value<std::string>(), "The number of MAE");

    boost::program_options::variables_map vm;
    store(parse_command_line(argc, argv, opt), vm);
    notify(vm);

    if(vm.count("help"))
    {
        std::cout << opt << std::endl;
        std::exit(0);
    }

    if(!vm.count("network") || !vm.count("sample") || !vm.count("target") || !vm.count("log") || !vm.count("output"))
    {
        std::cout << "Need --network, --sample, --target, --log and --output" << std::endl;
        std::cout << opt << std::endl;
        std::exit(0);
    }

    return {
        vm["network"].as<std::string>(),
        vm["sample"].as<std::string>(),
        vm["target"].as<std::string>(),
        vm["log"].as<std::string>(),
        vm["output"].as<std::string>()
    };
}

struct logfile_t {
    struct learning_unit {
        struct pruning_unit {
            bool is_jm_pruned;
            bool is_km_pruned;
            double jm_similarity;
            double km_similarity;

            std::size_t j_cluster_size;
            std::size_t k_cluster_size;
            std::size_t m_cluster_size;

            double probability;
            bool is_pruned;
        };

        std::vector<pruning_unit> pruns;
        std::size_t pruning_num;
        double time;
    };

    std::vector<learning_unit> logs;

    template<class InputStream>
    void parse(InputStream& is)
    {
        std::string parsing_line;
        std::getline(is, parsing_line);

        // ���ʍs�̂Ă�
        while(parsing_line.compare(0,2,"--") == 0 || parsing_line == "Learning...")
            std::getline(is, parsing_line);

        // �����܂łԂ��
        logfile_t log;
        while(true)
        {
            if(parsing_line == "Learning is end") break;

            logfile_t::learning_unit lunit;

            // �}����񐔕\���܂Œׂ�
            while(!std::isdigit(parsing_line[0]))
            {
                logfile_t::learning_unit::pruning_unit punit;

                std::istringstream iss(parsing_line);
                std::tie(punit.is_jm_pruned, punit.jm_similarity) = parse_similarity(iss);
                std::tie(punit.is_km_pruned, punit.km_similarity) = parse_similarity(iss);
                std::tie(punit.j_cluster_size, punit.k_cluster_size, punit.m_cluster_size) = parse_cluster_size(iss);
                punit.probability = parse_probability(iss);

                std::string remain;
                std::getline(iss, remain);
                punit.is_pruned = !remain.empty();

                lunit.pruns.push_back(std::move(punit));

                std::getline(is, parsing_line);
            }

            lunit.pruning_num = std::stoull(parsing_line);
            std::getline(is, parsing_line); // empty line
            std::getline(is, parsing_line);

            if(parsing_line.compare(0, 9, "Learned: ") != 0)
                throw std::runtime_error("Error");

            std::string const time = parsing_line.substr(9, parsing_line.size() - 9 - 4);
            lunit.time = std::stod(time);
            std::getline(is, parsing_line);

            log.logs.push_back(std::move(lunit));
        }

        *this = std::move(log);
    }

private:
    template<class InputStream>
    std::tuple<bool, double> parse_similarity(InputStream& is)
    {
        bool const is_pruned = !(is.get() == '[');

        std::string similarity;
        std::getline(is, similarity, is_pruned ? ' ' : ']');

        is.get(); // ' '

        return std::make_tuple(is_pruned, std::stod(similarity));
    }

    template<class InputStream>
    std::tuple<std::size_t, std::size_t, std::size_t> parse_cluster_size(InputStream& is)
    {
        is.get(); // '('

        std::string first_size;
        std::getline(is, first_size, ' ');

        is.get(); // ','
        is.get(); // ' '

        std::string second_size;
        std::getline(is, second_size, ' ');

        is.get(); // ','
        is.get(); // ' '

        std::string third_size;
        std::getline(is, third_size, ')');

        is.get(); // ' '

        return std::make_tuple(std::stoull(first_size), std::stoull(second_size), std::stoull(third_size));
    }

    template<class InputStream>
    double parse_probability(InputStream& is)
    {
        std::string probability;
        std::getline(is, probability, ' ');

        is.get(); // ' '

        return std::stod(probability);
    }
};

struct combine_phase {
    std::vector<std::vector<bn::vertex_type>> clusters;
};

template<class Eval, template<class> class BetweenLearning, class PruningProbExpr>
class sshc_emulator {
public:
    using graph_t = bn::graph_t;
    using vertex_type = bn::vertex_type;
    using sampler = bn::sampler;
    using cluster_type = std::shared_ptr<std::vector<vertex_type>>;
    using similarity_type = std::tuple<std::tuple<cluster_type, cluster_type>, double, int>;
    using Similarity = bn::evaluation::mutual_information;

    sshc_emulator(graph_t const& teacher_graph, sampler const& sampling, logfile_t::learning_unit const& log)
        : teacher_graph_(teacher_graph), sampling_(sampling), learning_machine_(sampling_), mutual_information_machine_(), engine_(bn::make_engine<std::mt19937>()), log_(log), log_it(log_.pruns.cbegin())
    {
    }

    // �K�w�I�N���X�^�����O�y�ъm���I�}�����p�����i�K�I�\���w�K�@�̎��s
    // graph: �O���t�\��(�ǂ�ȍ\���������Ă����Ƃ��Ă��C�N���A�����)
    // alpha: �}���肪���s�����m���Ɋւ���W��
    double operator()(graph_t& graph, double const alpha)
    {
        // graph�̏�����
        graph.erase_all_edge();

        // �����N���X�^�Ə����ގ��x��������
        initial_clustering(graph.vertex_list()); // �����N���X�^
        std::cout << "initialized clusters" << std::endl;
        initial_similarities();                  // �����ގ��x
        std::cout << "initialized similarities" << std::endl;

        // �N���X�^�Ԋw�K(����)
        auto const score = learning_between_clusters(graph, alpha);
        std::cout << "ended clusterings" << std::endl;
        return score;
    }

    std::vector<combine_phase> record()
    {
        return record_;
    }

private:
    // �����N���X�^�����O���s���C�N���X�^�����O���ʂ������o�ϐ�clusters_�Ɋi�[����
    // ��1����: �N���X�^�����O�Ώۂ̃m�[�h�W��
    void initial_clustering(std::vector<vertex_type> const& nodes)
    {
        // �N���X�^�W��������
        clusters_.clear();
        clusters_.reserve(nodes.size());

        // 1�m�[�h1�N���X�^
        for(auto const& node : nodes)
        {
            auto cluster = std::make_shared<cluster_type::element_type>();
            cluster->push_back(node);
            clusters_.push_back(std::move(cluster));
        }
    }

    // �����ގ��x�v�Z���s���C�ގ��x�������o�ϐ�similarities_�Ɋi�[����
    void initial_similarities()
    {
        // �ގ��x�W��������
        similarities_.clear();
        similarities_.reserve(clusters_.size() * clusters_.size());

        // �ގ��x���Ϗ�����
        average_similar_ = 0.0;
        auto const max_edge_num = clusters_.size() * (clusters_.size() - 1) / 2;

        // �S�N���X�^�y�A�ɂ��āC�ގ��x�v�Z
        for(std::size_t i = 0; i < clusters_.size(); ++i)
        {
            for(std::size_t j = i + 1; j < clusters_.size(); ++j)
            {
                auto const& i_cluster = clusters_[i];
                auto const& j_cluster = clusters_[j];
                assert(i_cluster->size() == 1 && j_cluster->size() == 1); // ������Ԃ�1�m�[�h�ł���͂�������

                auto&& similarity = make_similarity_tuple(i_cluster, j_cluster);
                average_similar_ += std::get<1>(similarity) / max_edge_num;
                similarities_.push_back(std::move(similarity));
            }
        }
    }

    // �w�肵��similarity��cluster���֗^���Ă��邩(cluster�Ɋւ���ގ��x��)�ǂ�����Ԃ�
    bool is_related(similarity_type const& similarity, cluster_type const& cluster)
    {
        auto const& connection = std::get<0>(similarity);
        return std::get<0>(connection) == cluster || std::get<1>(connection) == cluster;
    }

    // �w�肵��similarity��lhs��rhs�Ɋւ���ގ��x���ǂ�����Ԃ�
    bool is_connected(similarity_type const& similarity, cluster_type const& lhs, cluster_type const& rhs)
    {
        auto const& connection = std::get<0>(similarity);
        return std::get<0>(connection) == std::min(lhs, rhs) && std::get<1>(connection) == std::max(lhs, rhs);
    }

    // ������2�̃N���X�^��1�̃N���X�^�Ɍ�������
    // clusters_�ɒǉ�����Ă���΁Cclusters_����폜����(����p)
    cluster_type combine_clusters(cluster_type const& lhs, cluster_type const& rhs)
    {
        // �����N���X�^
        auto new_cluster = std::make_shared<cluster_type::element_type>();
        new_cluster->reserve(lhs->size() + rhs->size());
        new_cluster->insert(new_cluster->end(), lhs->cbegin(), lhs->cend());
        new_cluster->insert(new_cluster->end(), rhs->cbegin(), rhs->cend());

        // �O�̃N���X�^������
        clusters_.erase(std::find(clusters_.begin(), clusters_.end(), lhs));
        clusters_.erase(std::find(clusters_.begin(), clusters_.end(), rhs));

        return new_cluster;
    }

    // �����o�ϐ�similarities_����ł����ʂ̍������̂����o���C����ׂɐe�q�����肷��
    similarity_type most_similarity()
    {
        // �ł����Ă���N���X�^��
        auto const most_similar = std::max_element(
            similarities_.begin(), similarities_.end(),
            [](similarity_type const& lhs, similarity_type const& rhs)
            {
                return std::get<2>(lhs) < std::get<2>(rhs)
                    || !(std::get<2>(rhs) < std::get<2>(lhs)) && std::get<1>(lhs) < std::get<1>(rhs);
            });

        // �R�s�[���Č��̃N���X�^�Ԃ�����
        auto result = *most_similar;
        similarities_.erase(most_similar);

        auto pair = std::get<0>(result);
        auto const is_equal_until_six_dp = [](double const lhs, double const rhs)
            {
                return std::abs(lhs - rhs) <= 0.000001;
            };
        auto const is_random_order =
            [](cluster_type const& lhs_first, cluster_type const& lhs_second, cluster_type const& rhs_first, cluster_type const& rhs_second)
            {
                return (lhs_first == rhs_first && lhs_second == rhs_second)
                    || (lhs_first == rhs_second && lhs_second == rhs_first);
            };

        //auto local_log_it = log_it;
        //for(std::size_t i = 0; i < clusters_.size(); ++i, ++local_log_it)
        //{
        //    if(is_equal_until_six_dp(local_log_it->jm_similarity, local_log_it->km_similarity))
        //    {
        //        // ���f������
        //        continue;
        //    }
        //    else
        //    {
        //        // clusters[i]�Ƃ̗ގ��x�����o��
        //        std::tuple<similarity_type, similarity_type> forward_similarity;
        //        for(auto const& similarity : similarities_)
        //        {
        //            if(is_random_order(
        //                std::get<0>(std::get<0>(similarity)), std::get<1>(std::get<0>(similarity)),
        //                clusters_[i], std::get<0>(std::get<0>(result))))
        //            {
        //                std::get<0>(forward_similarity) = similarity;
        //            }
        //            else if(is_random_order(
        //                std::get<0>(std::get<0>(similarity)), std::get<1>(std::get<0>(similarity)),
        //                clusters_[i], std::get<1>(std::get<0>(result))))
        //            {
        //                std::get<1>(forward_similarity) = similarity;
        //            }
        //        }

        //        if(clusters_[i]->size() == local_log_it->m_cluster_size
        //            && std::get<0>(std::get<0>(result))->size() == local_log_it->j_cluster_size
        //            && std::get<1>(std::get<0>(result))->size() == local_log_it->k_cluster_size
        //            && is_equal_until_six_dp(std::get<1>(std::get<0>(forward_similarity)), local_log_it->jm_similarity)
        //            && is_equal_until_six_dp(std::get<1>(std::get<1>(forward_similarity)), local_log_it->km_similarity)
        //            )
        //        {
        //            std::cout << "forward\n";
        //            break;
        //        }

        //        if(clusters_[i]->size() == local_log_it->m_cluster_size
        //            && std::get<1>(std::get<0>(result))->size() == local_log_it->j_cluster_size
        //            && std::get<0>(std::get<0>(result))->size() == local_log_it->k_cluster_size
        //            && is_equal_until_six_dp(std::get<1>(std::get<1>(forward_similarity)), local_log_it->jm_similarity)
        //            && is_equal_until_six_dp(std::get<1>(std::get<0>(forward_similarity)), local_log_it->km_similarity)
        //            )
        //        {
        //            std::cout << "reverse\n";
        //            std::swap(std::get<0>(pair), std::get<1>(pair));
        //            break;
        //        }
        //    }
        //}

        return result;
    }

    std::tuple<cluster_type, cluster_type> make_cluster_tuple(cluster_type const& lhs, cluster_type const& rhs)
    {
        // �A�h���X���������N���X�^���ɂ��ĕԂ�
        return (lhs < rhs) ? std::make_tuple(lhs, rhs)
                           : std::make_tuple(rhs, lhs);
    }

    // �^����ꂽ2�̃N���X�^�Ԃ̗ގ��x�����߁Csimilarity_type�Ƃ��ĕԂ�
    similarity_type make_similarity_tuple(cluster_type const& lhs, cluster_type const& rhs)
    {
        // 2�N���X�^�Ԃ̃m�[�h�̂��ꂼ��̑g�ݍ��킹��
        auto const combination_num = lhs->size() * rhs->size();

        // �ގ��x�v�Z
        double value = 0;
        for(auto const& lhs_nodes : *lhs)
        {
            for(auto const& rhs_nodes : *rhs)
            {
                // ���Ŋ����đ���(����)
                value += mutual_information_machine_(sampling_, lhs_nodes, rhs_nodes) / combination_num;
            }
        }

        return std::make_tuple(make_cluster_tuple(lhs, rhs), value, 1);
    }

    // �N���X�^�Ԋw�K���s��
    // graph, alpha: operator()�Q��
    double learning_between_clusters(graph_t& graph, double const alpha)
    {
        double score = std::numeric_limits<double>::max();

        while(clusters_.size() != 1 && !similarities_.empty())
        {
            record_clusters();
            std::cout << "remain clusters: " << clusters_.size() << std::endl;

            //
            // �K�w�I�\���w�K ����
            //

            // �����Ώۂ𓾂�
            auto const similarity_target = most_similarity();
            if(std::get<2>(similarity_target) != 1) break;

            // learning
            auto const combine_target = std::get<0>(similarity_target);
            auto const parent = std::get<0>(combine_target);
            auto const child  = std::get<1>(combine_target);

            //!! TODO:
            score = learning_machine_.learn_with_hint(graph, teacher_graph_, *parent, *child);

            // �N���X�^����
            auto combined_cluster = combine_clusters(parent, child);

            //
            // �m���I�}���� ����
            //
            stochastic_pruning(alpha, combined_cluster, similarity_target);
            clusters_.push_back(combined_cluster);
        }

        record_clusters();
        return score;
    }

    // �m���I�}������s��
    // alpha: operator()�ɏ�����
    // new_cluster: ������̃N���X�^������
    // old_connection: �����O��2�N���X�^�Ԃ�similarity_type������
    void stochastic_pruning(
        double const alpha,
        cluster_type const& new_cluster, similarity_type const& old_similarity
        )
    {
        // Initialize
        std::unordered_map<cluster_type, std::tuple<double, std::vector<similarity_type>>> next_similarity;
        for(auto const& cluster : clusters_)
            next_similarity[cluster] = std::make_tuple(0.0, std::vector<similarity_type>());

        auto const is_equal_until_six_dp = [](double const lhs, double const rhs)
            {
                return std::abs(lhs - rhs) <= 0.000001;
            };

        // ���N���X�^�ގ��x���V�N���X�^�ގ��x���Z�o���C��������
        for(auto it = similarities_.begin(); it != similarities_.end();)
        {
            auto const& connection = std::get<0>(*it);
            auto const& old_connection = std::get<0>(old_similarity);
            if(std::get<0>(connection) == std::get<0>(old_connection) || std::get<0>(connection) == std::get<1>(old_connection))
            {
                auto const& old_cluster = std::get<0>(connection);
                auto const& target_cluster = std::get<1>(connection);
                auto const similarity = std::get<1>(*it);

                // �i�[�ƍ폜
                auto& container = next_similarity[target_cluster];
                std::get<0>(container) += similarity * old_cluster->size() / new_cluster->size();
                std::get<1>(container).push_back(*it);
                it = similarities_.erase(it);
            }
            else if(std::get<1>(connection) == std::get<0>(old_connection) || std::get<1>(connection) == std::get<1>(old_connection))
            {
                auto const& old_cluster = std::get<1>(connection);
                auto const& target_cluster = std::get<0>(connection);
                auto const similarity = std::get<1>(*it);

                // �i�[�ƍ폜
                auto& container = next_similarity[target_cluster];
                std::get<0>(container) += similarity * old_cluster->size() / new_cluster->size();
                std::get<1>(container).push_back(*it);
                it = similarities_.erase(it);
            }
            else ++it;
        }

        // �V�N���X�^�ގ��x���}��������s����
        for(std::size_t i = 0, end = clusters_.size(); i < end; ++i, ++log_it)
        {
            auto const& prunning_target = next_similarity[clusters_[i]];

            unsigned char flag = 0;
            flag |= std::get<2>(std::get<1>(prunning_target)[0]);
            flag |= std::get<2>(std::get<1>(prunning_target)[1]) * 2;

            double const prunning_probability =
                (flag == 3) ? pruning_probability_.p1(alpha, average_similar_, std::get<1>(old_similarity), std::get<0>(prunning_target), std::get<1>(std::get<1>(prunning_target)[0]), std::get<1>(std::get<1>(prunning_target)[1])) :
                (flag == 2) ? pruning_probability_.p2(alpha, average_similar_, std::get<1>(old_similarity), std::get<0>(prunning_target), std::get<1>(std::get<1>(prunning_target)[1]), std::get<1>(std::get<1>(prunning_target)[0])) :
                (flag == 1) ? pruning_probability_.p2(alpha, average_similar_, std::get<1>(old_similarity), std::get<0>(prunning_target), std::get<1>(std::get<1>(prunning_target)[0]), std::get<1>(std::get<1>(prunning_target)[1])) :
                              pruning_probability_.p3(alpha, average_similar_, std::get<1>(old_similarity), std::get<0>(prunning_target), std::get<1>(std::get<1>(prunning_target)[0]), std::get<1>(std::get<1>(prunning_target)[1]));

            // �m���ɂ��Cprobability�̊m���Ŏ}����
            if(!is_equal_until_six_dp(log_it->probability, prunning_probability))
                throw std::runtime_error("Oops");

            similarities_.emplace_back(make_cluster_tuple(clusters_[i], new_cluster), std::get<0>(prunning_target), log_it->is_pruned ? 0 : 1);
        }
    }

    void record_clusters()
    {
        combine_phase now;
        for(auto const& cluster : clusters_)
        {
            auto const raw_cluster = *cluster;
            now.clusters.push_back(raw_cluster);
        }

        record_.push_back(std::move(now));
    }

    graph_t const& teacher_graph_;
    sampler const& sampling_;
    BetweenLearning<Eval> learning_machine_;
    bn::evaluation::mutual_information mutual_information_machine_;
    std::mt19937 engine_;

    PruningProbExpr pruning_probability_;
    logfile_t::learning_unit const log_;
    std::vector<logfile_t::learning_unit::pruning_unit>::const_iterator log_it;

    std::vector<cluster_type> clusters_;
    std::vector<similarity_type> similarities_;
    double average_similar_;

    std::vector<combine_phase> record_;
};

template<class Eval>
class from_teacher {
public:
    using graph_t = bn::graph_t;
    using vertex_type = bn::vertex_type;

    from_teacher(bn::sampler const& sampling)
    {
    }

    double learn_with_hint(graph_t& graph, graph_t const& teacher_graph, std::vector<vertex_type> parent_nodes, std::vector<vertex_type> child_nodes)
    {
        auto const& vertexes = graph.vertex_list();
        auto const& teacher_vertexes = teacher_graph.vertex_list();

        // �e���Ǝq����S����
        for(auto const& child : child_nodes)
        {
            auto const child_index = std::distance(vertexes.begin(), std::find(vertexes.begin(), vertexes.end(), child));
            for(auto const& parent : parent_nodes)
            {
                auto const parent_index = std::distance(vertexes.begin(), std::find(vertexes.begin(), vertexes.end(), parent));

                for(auto const& teacher_edge : teacher_graph.edge_list())
                {
                    auto const& source = teacher_graph.source(teacher_edge);
                    auto const& target = teacher_graph.target(teacher_edge);

                    // �����N�͂��邩�ǂ���
                    if(source == teacher_vertexes[parent_index] && target == teacher_vertexes[child_index])
                    {
                        graph.add_edge(vertexes[parent_index], vertexes[child_index]);
                    }
                }
            }
        }

        return std::numeric_limits<double>::quiet_NaN();
    }
};

#include "./settings.hpp"

std::vector<std::vector<combine_phase>> process_calulate_mae(bn::graph_t& graph, bn::database_t const& database, bn::sampler const& sampler, logfile_t const& log, boost::filesystem::path const& path)
{
    // ��ƃp�X
    boost::filesystem::path const working_directory = path;
    boost::filesystem::path const result_path = path / "result.csv";

    // result.csv���J���C��́C����mae�v�Z
    boost::filesystem::ifstream res_ifs(result_path);
    std::stringstream str;
    std::string tmp;

    // 1�s�ړǂݔ�΂�
    std::getline(res_ifs, tmp);
    {
        std::vector<std::string> line;
        boost::algorithm::split(line, tmp, boost::is_any_of(","));
        
        line.resize(9);
        line[8] = "Pruning";
        str << boost::algorithm::join(line, ",") << "\n";
    }


    std::size_t counter = 0;
    std::size_t all_pruning = 0;

    std::vector<std::vector<combine_phase>> all_results;
    while(std::getline(res_ifs, tmp)) // 1�s���ǂݍ���
    {
        // �J���}��؂�
        std::vector<std::string> line;
        boost::algorithm::split(line, tmp, boost::is_any_of(","));

        if(line[0] == "Ave.")
        {   
            // �}����񐔂̍X�V
            line.resize(9);
            line[8] = std::to_string(static_cast<double>(all_pruning) / counter);
            str << boost::algorithm::join(line, ",") << "\n";

            break;
        }

        // �O���t���R�s�[
        auto teacher_graph = graph;
        teacher_graph.erase_all_edge();

        // �w�K�ς݃O���t�iteacher_graph�j���쐬
        auto const graph_path = working_directory / ("graph" + line[0] + ".csv");
        boost::filesystem::ifstream graph_ifs(graph_path);
        bn::serializer::csv().load(graph_ifs, teacher_graph);
        graph_ifs.close();

        // �G�~�����[�g
        auto const method_name = working_directory.stem().string();
        auto it = std::find_if(algorithms.begin(), algorithms.end(), [&method_name](algorithm_holder const& algo){ return algo.name == method_name; });

        auto result = it->function(graph, teacher_graph, sampler, log.logs[std::stoi(line[0])]);
        all_results.push_back(std::move(result));

        // �}����񐔂̍X�V
        line.resize(9);
        line[8] = std::to_string(log.logs[std::stoi(line[0])].pruning_num);

        // �}����񐔂̃J�E���g
        all_pruning += log.logs[std::stoi(line[0])].pruning_num;
        ++counter;

        str << boost::algorithm::join(line, ",") << "\n";
    }
    res_ifs.close();

    // result.csv����������
    boost::filesystem::ofstream res_ofs(result_path);
    res_ofs << str.rdbuf();
    res_ofs.close();

    return all_results;
}

int main(int argc, char* argv[])
{
    auto const command_line = process_command_line(argc, argv);

    std::cout << "Starting..." << std::endl;
    std::cout << "Network: " << command_line.network << std::endl;
    std::cout << "Target: " << command_line.target << std::endl;

    // �O���t�t�@�C�����J����graph_data�ɓ���
    std::string const graph_data{std::istreambuf_iterator<char>(boost::filesystem::ifstream(command_line.network)), std::istreambuf_iterator<char>()};
    std::cout << "Loaded Graph: Length = " << graph_data.size() << std::endl;

    // graph_data���O���t�p�[�X
    bn::graph_t graph;
    bn::database_t data;
    std::tie(graph, data) = bn::serializer::bif().parse(graph_data.cbegin(), graph_data.cend());
    std::cout << "Parsed Graph: Num of Node = " << graph.vertex_list().size() << std::endl;

    // �T���v��
    bn::sampler sampler(command_line.sample.string());
    sampler.load_sample(graph.vertex_list());
    sampler.make_cpt(graph);
    std::cout << "Loaded Sample: Num of Sample = " << sampler.sampling_size() << std::endl;

    // ���O�t�@�C����ǂ�
    logfile_t log;
    log.parse(boost::filesystem::ifstream(command_line.log));
    std::cout << "Loaded Log: Num of Log = " << log.logs.size() << std::endl;

    // Run!
    auto const results = process_calulate_mae(graph, data, sampler, log, command_line.target);

    
    for(std::size_t i = 0; i < /*results.size()*/ 1; ++i)
    {
        std::vector<std::pair<std::vector<bn::vertex_type>, std::string>> clusters_dic;
        for(auto const& vertex : graph.vertex_list())
        {
            std::vector<bn::vertex_type> cluster{vertex};
            clusters_dic.emplace_back(std::move(cluster), data.node_name[vertex->id]);
        }

        for(auto const& phase : results[i])
        {
        }
    }



    // Output TODO:
    boost::filesystem::ofstream ofs(command_line.output);
    for(std::size_t i = 0; i < results.size(); ++i)
    {
        ofs << "---------- graph" << std::to_string(i) << ".csv ----------\n";
        for(auto const& phase : results[i])
        {
            for(auto cluster : phase.clusters)
            {
                std::sort(cluster.begin(), cluster.end(), [](bn::vertex_type const& lhs, bn::vertex_type const& rhs) { return lhs->id < rhs->id; });
                for(std::size_t j = 0; j < cluster.size(); ++j)
                {
                    if(j == 0) ofs << data.node_name.at(cluster[j]->id);
                    else       ofs << " " << data.node_name.at(cluster[j]->id);
                }
                ofs << "\n";
            }
            ofs << "\n";
        }
    }
    ofs.close();
}
