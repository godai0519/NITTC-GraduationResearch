#ifndef GRADRESEARCH_EQLISTMANAGER_HPP
#define GRADRESEARCH_EQLISTMANAGER_HPP

#include <unordered_map>
#include <utility>
#include <boost/filesystem/path.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <bayesian/graph.hpp>

namespace gr {

struct calculate_target {
    typedef std::pair<bn::vertex_type, std::size_t> query_type;
    typedef std::unordered_map<bn::vertex_type, std::size_t> evidence_type;

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
    InputStream& load_calculate_target(InputStream& ist, std::vector<bn::vertex_type> const& nodes)
    {
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

    query_type query;
    evidence_type evidence;
    double inference;
};

class eqlist_manager {
public:
    using query_type = calculate_target::query_type;
    using evidence_type = calculate_target::evidence_type;

    explicit eqlist_manager();
    explicit eqlist_manager(std::size_t const size);
    void load(boost::filesystem::path const& path, std::vector<bn::vertex_type> const& nodes, std::size_t const size);
    
    void add(query_type const& query, evidence_type const& evidence, double const inference);
    void remove(std::size_t const index);
    void remove_all();
    void reserve(std::size_t const size);


    std::size_t size() const;
    std::vector<calculate_target> const& target() const;

private:
    std::vector<calculate_target> targets_;
};

} // namespace gr

#endif // #ifndef GRADRESEARCH_EQLISTMANAGER_HPP
