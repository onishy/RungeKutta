#pragma once

#include <functional>
#include <Eigen/Geometry>


// N変数runge-kutta
template <unsigned int N>
class RungeKutta
{
public:
    using VectorNf = Eigen::Matrix<float, N, 1>;

    RungeKutta(std::function<VectorNf(float, const VectorNf &)> f)
        : m_func(f) {}

    VectorNf run(float t, const VectorNf &in, float h)
    {
        VectorNf k1, k2, k3, k4;
        k1 = m_func(t, in);
        k2 = m_func(t + h / 2, in + (h / 2) * k1);
        k3 = m_func(t + h / 2, in + (h / 2) * k2);
        k4 = m_func(t + h, in + h * k3);

        return in + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    }

private:
    std::function<VectorNf(float, const VectorNf &)> m_func;
};