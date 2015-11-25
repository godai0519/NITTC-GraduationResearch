#ifndef PRE_EXP_ALGORITHMS_HPP
#define PRE_EXP_ALGORITHMS_HPP

#include <random>
#include <chrono>
#include <tuple>
#include <boost/timer/timer.hpp>
#include <bayesian/graph.hpp>
#include <bayesian/sampler.hpp>

struct result_t {
    bn::graph_t graph;
    double      score;
    double      time;
    double      mae;
    std::size_t disappeared_link;
    std::size_t appeared_link;
    std::size_t reversed_link;
    double      change_mi;
};

result_t learning(
    bn::graph_t const& teacher_graph,
    bn::sampler const& sampler,
    std::function<double(bn::graph_t& graph, bn::sampler const& sampler)> func
    );

#endif
