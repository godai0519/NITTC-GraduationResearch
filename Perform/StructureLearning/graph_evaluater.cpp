#include <algorithm>
#include <vector>
#include "graph_evaluater.hpp"

std::size_t count_disappeared_link(bn::graph_t const& teacher, bn::graph_t const& target)
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
                return teacher.source(teacher_edge) == target.source(target_edge)
                    && teacher.target(teacher_edge) == target.target(target_edge);
            });

        if(it == target_edges.end()) ++counter;
    }

    return counter;
}

std::size_t count_appeared_link(bn::graph_t const& teacher, bn::graph_t const& target)
{
    auto const& teacher_edges = teacher.edge_list();
    auto target_edges = target.edge_list();

    for(auto const& teacher_edge : teacher_edges)
    {
        auto it = std::find_if(
            target_edges.begin(), target_edges.end(),
            [&teacher_edge, &teacher, &target](bn::edge_type const& target_edge)
            {
                return teacher.source(teacher_edge) == target.source(target_edge)
                    && teacher.target(teacher_edge) == target.target(target_edge);
            });

        if(it != target_edges.end()) target_edges.erase(it);
    }

    return target_edges.size();
}
