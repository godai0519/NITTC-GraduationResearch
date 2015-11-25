#include <boost/timer/timer.hpp>
#include "graph_evaluater.hpp"
#include "algorithms.hpp"

result_t learning(
    bn::graph_t const& teacher_graph,
    bn::sampler const& sampler,
    std::function<double(bn::graph_t& graph, bn::sampler const& sampler)> func)
{
    // グラフの辺を全て削除する
    auto graph = teacher_graph;
    graph.erase_all_edge();

    // 学習
    boost::timer::cpu_timer timer;
    auto const score = func(graph, sampler);

    // 計測値の取得
    timer.stop();
    auto const elapsed = timer.elapsed();

    // 消失リンク数の数え上げ
    auto const disappeared_link = count_disappeared_link(teacher_graph, graph);

    // 増加リンク数の数え上げ
    auto const appeared_link = count_appeared_link(teacher_graph, graph);

    // 反転リンク数の数え上げ
    auto const reversed_link = count_reversed_link(teacher_graph, graph);

    return result_t{
        std::move(graph),
        score,
        static_cast<double>(elapsed.user) * 1.0e-9, // from nanoseconds to seconds
        std::numeric_limits<double>::quiet_NaN(),
        disappeared_link,
        appeared_link,
        reversed_link,
        std::numeric_limits<double>::quiet_NaN()
    };
}
