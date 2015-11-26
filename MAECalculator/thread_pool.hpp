#ifndef GRADRESEARCH_THREADPOOL_HPP
#define GRADRESEARCH_THREADPOOL_HPP

#include <functional>
#include <memory>
#include <boost/asio/io_service.hpp>
#include <boost/thread/thread.hpp>

namespace gr {

class thread_pool {
public:
    typedef std::function<void()> work_type;
    
    explicit thread_pool(std::size_t const thread_num);
    thread_pool(boost::asio::io_service& io_service, std::size_t const thread_num);
    ~thread_pool();

    void post(work_type work);
    void stop();

private:
    static boost::asio::io_service& build_io_service();
    void init_thread_group(std::size_t const thread_num);

    boost::asio::io_service& io_service_;
    std::unique_ptr<boost::asio::io_service::work> worker_;
    boost::thread_group tgroup_;

};

} // namespace gr

#endif // #ifndef GRADRESEARCH_THREADPOOL_HPP
