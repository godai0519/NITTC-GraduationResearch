#include <bayesian/evaluation/aic.hpp>
#include <bayesian/learning/brute_force.hpp>
#include <bayesian/learning/greedy.hpp>
#include <bayesian/learning/simulated_annealing.hpp>
#include <bayesian/learning/stepwise_structure.hpp>
#include <bayesian/learning/stepwise_structure_hc.hpp>

std::size_t const iteration_num = 2; // 10
std::size_t const MAE_REPEAT_NUM = 10; // 100

using EvaluationAlgorithm = bn::evaluation::aic;

struct algorithm_holder {
    std::string name;
    std::function<double(bn::graph_t& graph, bn::sampler const& sampler)> function;
};

std::vector<algorithm_holder> const algorithms = {
    //{
    //    "greedy",
    //    [](bn::graph_t& graph, bn::sampler const& sampler)
    //    {
    //        bn::learning::greedy<EvaluationAlgorithm> greedy(sampler);
    //        return greedy(graph);
    //    }
    //},
    ///*{
    //    "simulated_annealing_1000to1_98",
    //    [](bn::graph_t& graph, bn::sampler const& sampler)
    //    {
    //        bn::learning::simulated_annealing<EvaluationAlgorithm> sa(sampler);
    //        return sa(graph, 1000, 0.1, 0.98);
    //    }
    //},
    //{
    //    "stepwise_structure_greedy_greedy",
    //    [](bn::graph_t& graph, bn::sampler const& sampler)
    //    {
    //        bn::learning::stepwise_structure<EvaluationAlgorithm, bn::learning::greedy, bn::learning::greedy> ss(sampler);
    //        return ss(graph, 3);
    //    }
    //},*/
    {
        "stepwise_structure_bf_greedy",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure<EvaluationAlgorithm, bn::learning::brute_force, bn::learning::greedy> ss(sampler);
            return ss(graph, 3);
        }
    },
    {
        "stepwise_structure_hc_00",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy> sshc(sampler);
            return sshc(graph, 0.0);
        }
    },
    {
        "stepwise_structure_hc_05",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy> sshc(sampler);
            return sshc(graph, 0.05);
        }
    },
    {
        "stepwise_structure_hc_10",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy> sshc(sampler);
            return sshc(graph, 0.1);
        }
    },
    {
        "stepwise_structure_hc_20",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy> sshc(sampler);
            return sshc(graph, 0.2);
        }
    },
    {
        "stepwise_structure_hc_30",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy> sshc(sampler);
            return sshc(graph, 0.3);
        }
    },
    {
        "stepwise_structure_hc_40",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy> sshc(sampler);
            return sshc(graph, 0.4);
        }
    }
};
