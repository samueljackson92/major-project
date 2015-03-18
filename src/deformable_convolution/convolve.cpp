#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numeric.hpp>
#include <iostream>

using namespace boost::python;

char const* convolve(boost::python::numeric::array& a)
{
    // boost::python::numeric::array a = data;
    tuple shape = extract<tuple>(a.attr("shape"));
    int x = extract<int>(shape[0]);
    std::cout << x << std::endl;
    return "hello, world";
}

BOOST_PYTHON_MODULE(deformable_convolution)
{
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    def("convolve", convolve);
}
