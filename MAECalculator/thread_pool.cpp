#include <boost/bind.hpp>
#include "thread_pool.hpp"

namespace gr {

thread_pool::thread_pool(std::size_t const thread_num)
    : io_service_(build_io_service())
    , worker_(std::make_unique<boost::asio::io_service::work>(io_service_))
{
    init_thread_group(thread_num);
}

thread_pool::thread_pool(boost::asio::io_service& io_service, std::size_t const thread_num)
    : io_service_(io_service)
    , worker_(std::make_unique<boost::asio::io_service::work>(io_service_))
{
    init_thread_group(thread_num);
}

thread_pool::~thread_pool()
{
    worker_.reset();
    tgroup_.join_all();
}

void thread_pool::post(work_type work)
{
    io_service_.post(work);
}

void thread_pool::stop()
{
    worker_.reset();
    tgroup_.interrupt_all();
    io_service_.reset();
}

boost::asio::io_service& thread_pool::build_io_service()
{
    static boost::asio::io_service io_service;
    return io_service;
}

void thread_pool::init_thread_group(std::size_t const thread_num)
{
    for(std::size_t i = 0; i < thread_num; ++i)
        tgroup_.create_thread(boost::bind(&boost::asio::io_service::run, &io_service_));
}

} // namespace gr
