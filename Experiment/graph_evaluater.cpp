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
