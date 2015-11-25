#ifndef PRE_EXP_GRAPH_EVALUATER_HPP
#define PRE_EXP_GRAPH_EVALUATER_HPP

#include <bayesian/graph.hpp>

std::size_t count_disappeared_link(bn::graph_t const& teacher, bn::graph_t const& target);
std::size_t count_appeared_link(bn::graph_t const& teacher, bn::graph_t const& target);

#endif
