#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include "eqlist_manager.hpp"

namespace gr {
    
eqlist_manager::eqlist_manager() = default;
eqlist_manager::eqlist_manager(std::size_t const size)
{
    reserve(size);
}

void eqlist_manager::load(boost::filesystem::path const& path, std::vector<bn::vertex_type> const& nodes, std::size_t const size)
{
    reserve(size);

    boost::filesystem::ifstream ifs(path);
    for(int i = 0; i < size; ++i)
    {
        calculate_target t;
        t.load_calculate_target(ifs, nodes);
        targets_.push_back(t);
    }
    ifs.close();
}
    
void eqlist_manager::add(query_type const& query, evidence_type const& evidence, double const inference)
{
    targets_.push_back(calculate_target{query, evidence, inference});
}

void eqlist_manager::remove(std::size_t const index)
{
    targets_.erase(targets_.begin() + index);
}

void eqlist_manager::remove_all()
{
    targets_.clear();
}

void eqlist_manager::reserve(std::size_t const size)
{
    targets_.reserve(size);
}

std::size_t eqlist_manager::size() const
{
    return targets_.size();
}

std::vector<calculate_target> const& eqlist_manager::target() const
{
    return targets_;
}

} // namespace gr
