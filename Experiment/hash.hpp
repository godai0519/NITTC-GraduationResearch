#ifndef PRE_EXP_HASH_HPP
#define PRE_EXP_HASH_HPP

#include <functional>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/functional/hash.hpp>

namespace std 
{ 
    // boost::filesystem::path ‚ÌhashŠí
    template<> struct hash<boost::filesystem::path> 
    { 
        size_t operator()(const boost::filesystem::path& p) const
        {
            return boost::filesystem::hash_value(p);
        }
    }; 
}

#endif
