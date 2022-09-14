// Copyright (C) 2016-2020 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGS_H
#define LBFGS_H

//#define PRINT_MSG

#include <iomanip>  // for std::setprecision()


#include <Eigen/Core>
#include "LBFGSpp/Param.h"
#include "LBFGSpp/BFGSMat.h"
#include "LBFGSpp/LineSearchBacktracking.h"
#include "LBFGSpp/LineSearchBracketing.h"
#include "LBFGSpp/LineSearchNocedalWright.h"


namespace LBFGSpp {


///
/// L-BFGS solver for unconstrained numerical optimization
///
template < typename Scalar,
           template<class> class LineSearch = LineSearchBacktracking >
class LBFGSSolver
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Map<Vector> MapVec;

    const LBFGSParam<Scalar>& m_param;  // Parameters to control the LBFGS algorithm
    BFGSMat<Scalar>           m_bfgs;   // Approximation to the Hessian matrix
    Vector                    m_fx;     // History of the objective function values
    Vector                    m_xp;     // Old x
    Vector                    m_grad;   // New gradient
    Vector                    m_gradp;  // Old gradient
    Vector                    m_drt;    // Moving direction

    // Reset internal variables
    // n: dimension of the vector to be optimized
    inline void reset(int n)
    {
        const int m = m_param.m;
        m_bfgs.reset(n, m);
        m_xp.resize(n);
        m_grad.resize(n);
        m_gradp.resize(n);
        m_drt.resize(n);
        if(m_param.past > 0)
            m_fx.resize(m_param.past);
    }

public:
    ///
    /// Constructor for the L-BFGS solver.
    ///
    /// \param param An object of \ref LBFGSParam to store parameters for the
    ///        algorithm
    ///
    LBFGSSolver(const LBFGSParam<Scalar>& param) :
        m_param(param)
    {
        m_param.check_param();
    }

    ///
    /// Minimizing a multivariate function using the L-BFGS algorithm.
    /// Exceptions will be thrown if error occurs.
    ///
    /// \param f  A function object such that `f(x, grad)` returns the
    ///           objective function value at `x`, and overwrites `grad` with
    ///           the gradient.
    /// \param x  In: An initial guess of the optimal point. Out: The best point
    ///           found.
    /// \param fx Out: The objective function value at `x`.
    ///
    /// \return Number of iterations used.
    ///
    template <typename Foo>
    inline int minimize(Foo& f, Vector& x, Scalar& fx, int mpi_rank)
    {
        using std::abs;

        // Dimension of the vector
        const int n = x.size();
        reset(n);

        // The length of lag for objective function value to test convergence
        const int fpast = m_param.past;

        // printout linesearch condition
        if(mpi_rank == 0){
            std::cout << "lineserach condition : ";

            if(m_param.linesearch == 1){            
                std::cout << "Armijo." << std::endl;
            }
            else if(m_param.linesearch == 2){
                std::cout << "Wolfe." << std::endl;
            } else if (m_param.linesearch == 3){
                std::cout << "Strong Wolfe." << std::endl;
            } else {
                std::cout << "unknown. m_param.linesearch = " << m_param.linesearch << std::endl;
                exit(1);
            }
        }

        // Evaluate function and compute gradient
        fx = f(x, m_grad);
        Scalar gnorm = m_grad.norm();
        if(fpast > 0)
            m_fx[0] = fx;

        // Early exit if the initial x is already a minimizer
        if(gnorm <= m_param.epsilon || gnorm <= m_param.epsilon_rel * x.norm())
        {
            return 1;
        }

        // Initial direction
        m_drt.noalias() = -m_grad;
        // Initial step size
        // what is a good initial stepsize?
        Scalar step = Scalar(1) / m_drt.norm();

        // Number of iterations used
        int k = 1;
        for( ; ; )
        {
            // Save the curent x and gradient
            m_xp.noalias() = x;
            m_gradp.noalias() = m_grad;

            bool hess_reset = false;
            // Line search to update x, fx and gradient
            LineSearch<Scalar>::LineSearch(f, fx, x, m_grad, step, m_drt, m_xp, m_param, mpi_rank, hess_reset);
		
            if(hess_reset == true){
                m_bfgs.resetHess();
#ifdef PRINT_MSG
                if(mpi_rank == 0)
                    std::cout << "Hessian was reset." << std::endl;
#endif
            }

	        //std::cout << "step size = " << step << std::endl;
	        //std::cout << "gradient  = " << m_grad.transpose() << std::endl;
            // New gradient norm
            gnorm = m_grad.norm();

            /* std::cout << "k = " << k << ", m_fx : " << m_fx << std::endl;
            std::cout << "m_param.delta : " << m_param.delta << std::endl;
            std::cout << "abs(fxd - fx) : " << abs(m_fx[0] - fx) << ", other side : " << std::max(std::max(abs(fx), abs(m_fx[0])), Scalar(1)) << std::endl;
            */

            //a Convergence test -- gradient
            if(gnorm <= m_param.epsilon || gnorm <= m_param.epsilon_rel * x.norm())
            {	
//#ifdef PRINT_MSG
    		    if(mpi_rank == 0){
                    std::cout << "exited. epsilon = " << m_param.epsilon << ", epsilon_rel * xnorm = " << m_param.epsilon_rel * x.norm() << std::endl;
                    std::cout << "gnorm = " << gnorm << std::endl;
                }
//#endif
    		    return k;
            }
            // Convergence test -- objective function value
            if(fpast > 0)
            {
                const Scalar fxd = m_fx[k % fpast];
                // I CHANGED THIS at time to simply be <= m_param.delta !!
                if(k >= fpast && abs(fxd - fx) <= m_param.delta){ // * std::max(std::max(abs(fx), abs(fxd)), Scalar(1))){ // ){ 
//#ifdef PRINT_MSG
                    if(mpi_rank == 0)
                        std::cout << "convergence criterion met abs(f(x_k) - f(x_k-1)) = " << std::fixed << std::setprecision(8) << abs(fxd - fx) << std::endl;
//#endif
                    return k;
                }

                m_fx[k % fpast] = fx;
            }
            // Maximum number of iterations
            if(m_param.max_iterations != 0 && k >= m_param.max_iterations)
            {
                return k;
            }

            // Update s and y
            // s_{k+1} = x_{k+1} - x_k
            // y_{k+1} = g_{k+1} - g_k
            m_bfgs.add_correction(x - m_xp, m_grad - m_gradp, mpi_rank);

            // Recursive formula to compute d = -H * g
            // updating m_drt here
            m_bfgs.apply_Hv(m_grad, -Scalar(1), m_drt);

            // Reset step = 1.0 as initial guess for the next line search
            step = Scalar(1);
            k++;
        }

        return k;
    }
};


} // namespace LBFGSpp

#endif // LBFGS_H
