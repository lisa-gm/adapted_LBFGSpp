// Copyright (C) 2016-2020 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LINE_SEARCH_BACKTRACKING_H
#define LINE_SEARCH_BACKTRACKING_H

#include <Eigen/Core>
#include <stdexcept>  // std::runtime_error

//#define PRINT_MSG

namespace LBFGSpp {


///
/// The backtracking line search algorithm for L-BFGS. Mainly for internal use.
///
template <typename Scalar>
class LineSearchBacktracking
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

public:
    ///
    /// Line search by backtracking.
    ///
    /// \param f      A function object such that `f(x, grad)` returns the
    ///               objective function value at `x`, and overwrites `grad` with
    ///               the gradient.
    /// \param fx     In: The objective function value at the current point.
    ///               Out: The function value at the new point.
    /// \param x      Out: The new point moved to.
    /// \param grad   In: The current gradient vector. Out: The gradient at the
    ///               new point.
    /// \param step   In: The initial step length. Out: The calculated step length.
    /// \param drt    The current moving direction.
    /// \param xp     The current point.
    /// \param param  Parameters for the LBFGS algorithm
    ///
    template <typename Foo>
    static void LineSearch(Foo& f, Scalar& fx, Vector& x, Vector& grad,
                           Scalar& step,
                           Vector& drt, const Vector& xp,
                           const LBFGSParam<Scalar>& param,
                           int mpi_rank, bool& hess_reset)
    {
        // Decreasing and increasing factors
        const Scalar dec = 0.5;
        const Scalar inc = 2.1;

        // Check the value of step
        if(step <= Scalar(0))
            throw std::invalid_argument("'step' must be positive");

        // Save the function value at the current x
        const Scalar fx_init = fx;
        // Projection of gradient on the search direction
        const Scalar dg_init = grad.dot(drt);

#ifdef PRINT_MSG
        if(mpi_rank == 0){
            // Make sure d points to a descent direction
            std::cout << "dg_init = " << dg_init << ", norm(grad) = " << grad.norm()  << ", norm(drt) = " << drt.norm() << std::endl;
            std::cout << "grad    = " << grad.transpose() << std::endl;
            std::cout << "drt     = " << drt.transpose() << std::endl;
	    }
#endif


#ifdef PRINT_MSG
    if(mpi_rank == 0){
        std::cout << "\nNEW LINESEARCH" << std::endl;
        std::cout << "fx_init = " << fx_init << ", dg_init = " << dg_init << ", norm(drt) = " << drt.norm() << std::endl;
        //std::cout << "grad    = " << grad.transpose() << std::endl;
        std::cout << " new drt     = " << drt.transpose() << std::endl;
        std::cout << "dot(grad, drt) / (norm(grad)*norm(drt) = " << dg_init / (grad.norm()*drt.norm()) << std::endl;
    }
#endif

        // was originally clearly 0
	if(dg_init > 0){
        // ORIGINAL
        //throw std::logic_error("the moving direction increases the objective function value");

        // NEW
        // TODO: counter etc. so that this can't be done infinitely many times !!
        if(mpi_rank == 0){
            std::cout << "Set drt to equal negative gradient, see if that fixes problem." << std::endl;
        }

        drt = -1./grad.norm() * grad; // gradient normed to one, usually norm of drt smaller than 1
        step = 0.25;  // 0.5 

        // ask for reset of hessian
        hess_reset = true;

        // slightly perturb x
        //fx = f(x, grad);
        // Initial direction
        // Save the function value at the current x
        const Scalar fx_init = fx;
        // Projection of gradient on the search direction
        const Scalar dg_init = grad.dot(drt);
	}

        const Scalar test_decr = param.ftol * dg_init;
        Scalar width;

        int iter;
        for(iter = 0; iter < param.max_linesearch; iter++)
        {
            // x_{k+1} = x_k + step * d_k
            x.noalias() = xp + step * drt;
            // Evaluate this candidate
            fx = f(x, grad);

#ifdef PRINT_MSG
            if(mpi_rank == 0){
                std::cout << "iter = " << iter << ", step size = " << step << ", fx : " << std::right << std::fixed << fx << std::endl;   
                std::cout << "(fx - fx_init)/ dg_init = " << (fx - fx_init) / dg_init << std::endl;
            }
#endif

            if(fx > fx_init + step * test_decr || (fx != fx))
            {
                width = dec;
//#ifdef PRINT_MSG
                if(mpi_rank == 0){
                    std::cout << "linsearch iter = " << iter << ", first Wolfe condition not met." << std::endl;
                }
//#endif
            } else {
                // Armijo condition is met
                if(param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_ARMIJO){
#ifdef PRINT_MSG
                    if(mpi_rank == 0){
		 	            std::cout << "iter = " << iter << ", accepted step = " << step << std::endl;
                        std::cout << "theta   : " << std::right << std::fixed << x.transpose() << ",    f_theta : " << std::right << std::fixed << fx << std::endl;	    
		                std::cout << "grad    : " << grad.transpose() << "\n" << std::endl;
		    }
#endif
		    break;
		}

                const Scalar dg = grad.dot(drt);
                if(dg < param.wolfe * dg_init)
                {
                    width = inc;
//#ifdef PRINT_MSG
                    if(mpi_rank == 0){
                        std::cout << "linsearch iter = " << iter << ", second Wolfe condition not met." << std::endl;
			std::cout << "dg: " << dg << ", param.wolfe*dg_init: " << param.wolfe * dg_init << std::endl;
			//std::cout << "dg = " << dg << ", norm(grad)    : " << grad.norm() << "\n" << std::endl;
                        //std::cout << "dg / dg_init = " << dg / dg_init << std::endl;
                    }

//#endif

                } else {
                    // Regular Wolfe condition is met
                    if(param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE)
#ifdef PRINT_MSG       
                        if(mpi_rank == 0){
                            std::cout << "\niter = " << iter << ", accepted step = " << step << std::endl;
                            std::cout << "theta   : " << std::right << std::fixed << x.transpose() << ",    f_theta : " << std::right << std::fixed << fx << std::endl;
                            std::cout << "dg = " << dg << ", norm(grad)    : " << grad.norm() << std::endl;
                            std::cout << "norm(drt) = " << drt.norm() << ", grad    : " << grad.transpose() << std::endl;
                            std::cout << "dg / dg_init = " << dg / dg_init << ",   - dg_init / (norm(drt) * norm(grad)) = " << - dg / (drt.norm() * grad.norm()) << std::endl;
                        }
#endif
                        break;

                    if(dg > -param.wolfe * dg_init)
                    {
                        width = dec;
                    } else {
                        // Strong Wolfe condition is met
                        break;
                    }
                }
            }

            if(step < param.min_step)
                throw std::runtime_error("the line search step became smaller than the minimum value allowed");

            if(step > param.max_step)
                throw std::runtime_error("the line search step became larger than the maximum value allowed");

            step *= width;
        }

        if(iter >= param.max_linesearch){
#ifdef PRINT_MSG       
            if(mpi_rank == 0){
                std::cout << "maximum linesearch iterations reached. max = " << param.max_linesearch << std::endl;
                //std::cout << "exiting with stepsize = " << step << ", theta = " << x.transpose() << ", f_theta = " << fx << std::endl;
            	//std::cout << "accept current anyway. theta = " << x.transpose() << ", f_theta = " << fx << std::endl;
	       }
#endif
            throw std::runtime_error("the line search routine reached the maximum number of iterations");
	       //return;
        }
    }
};


} // namespace LBFGSpp

#endif // LINE_SEARCH_BACKTRACKING_H
