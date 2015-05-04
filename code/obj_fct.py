import numpy
import math

#Abstract Function Class
class Function(object):
    def __init__(self, size):
        """
        :param size: dimensions of domain
        :type size: int
        """
        self.start = None    # start point of iteration
        self.size = size

#Extended Rosenbrock Function
class Fct1(Function):
    def __init__(self, size):
        """
        :param size: dimensions of domain
        :type: int
        """
        super(Fct1, self).__init__(size)
        self.start = numpy.array([-1.2, 1]*(self.size/2))


    def kth_fct_val(self, k, x):
        """
        :param k: integer between 1 and function size, indicating kth sub-function
        :param x: a point in domain
        :return: kth function value at x
        """
        if(k & 0x1):
            return 10.0*(x[k] - x[k-1]**2)
        else:
            return 1.0 - x[k-2]

    def kth_fct_grad(self, k, x):
        """
        :param k: kth sub-function
        :param x: a point in domain
        :return: kth function's gradient at x
        """
        grad = numpy.zeros(self.size)
        if(k & 0x1):
            grad[k-1] = -20.0*x[k-1]
            grad[k] = 10.0
        else:
            grad[k-2] = -1.0

        return grad

    def val(self, x):
        """
        :param x: a point in domain
        :return: function's value at x
        """
        val = 0.0
        for i in xrange(1, self.size+1):
            val = val + self.kth_fct_val(i, x)**2

        return val

    def grad(self, x):
        """
        :param x: a point in domain
        :return: function's gradient at x
        """
        grad = numpy.zeros(self.size)
        for i in xrange(1, self.size+1):
            if(i & 0x1):
                grad[i-1] = -40.0 * self.kth_fct_val(i, x)*x[i-1] - 2.0 * self.kth_fct_val(i+1, x)
            else:
                grad[i-1] = 20.0 * self.kth_fct_val(i-1, x)

        return grad

    def hessian(self, x):
        """
        :param x: a point in domain
        :return: function's Hessian matrix at x
        """
        H = numpy.zeros(shape =(self.size, self.size))
        for i in xrange(1, self.size+1):
            if i & 0x1:
                H[i-1, i-1] = -40*self.kth_fct_val(i, x) + 800*x[i-1]**2 + 2
                if i < self.size:
                    H[i-1, i] = -400*x[i-1]
            else:
                H[i-1, i-1] = 200.0
                if i>1:
                    H[i-1, i-2] = -400*x[i-2]
        return H
    def get_answer(self):
        return numpy.zeros(self.size)+1.0

# Extended Powell Singular Function
class Fct2(Function):
    def __init__(self, size):
        """
        :param size: dimensions of domain
        """
        super(Fct2, self).__init__(size)
        self.start = numpy.array([3.0, -1.0, 0.0, 1.0]*(self.size/4))
        self.answer = numpy.zeros(self.size)

    def kth_fct_val(self, k, x):
        """
        :param k: an integer between 1 and function size, choosing kth sub-function
        :param x: a point in domain
        :return: kth sub-function's value at x
        """

        if k % 4 == 1:
            return x[k-1] + 10.0*x[k]
        elif k % 4 == 2:
            return math.sqrt(5.0)*(x[k] - x[k+1])
        elif k % 4 == 3:
            return (x[k-2] - 2.0*x[k-1])**2
        else:
            return math.sqrt(10.0)*(x[k-4]-x[k-1])**2

    def kth_fct_grad(self, k, x):
        """
        :param k: denote to choose kth sub-function
        :param x: a point in domain
        :return: kth sub-function's gradient at x
        """
        grad = numpy.zeros(self.size)
        if k%4 == 1:
            grad[k-1] = 1.0
            grad[k] = 10.0
        elif k%4 == 2:
            grad[k] = math.sqrt(5.0)
            grad[k+1] = -math.sqrt(5.0)
        elif k%4 == 3:
            grad[k-2] = 2.0 * (x[k-2] - 2.0*x[k-1])
            grad[k-1] = -4.0*(x[k-2] - 2.0*x[k-1])
        else:
            grad[k-4] = 2.0 * math.sqrt(10.0)*(x[k-4] - x[k-1])
            grad[k-1] = -2.0 * math.sqrt(10.0)*(x[k-4] - x[k-1])

        return grad

    def val(self, x):
        """
        :param x: a point in domain
        :return: function's value at x
        """
        val = 0.0
        for i in xrange (1, self.size+1):
            val = val + self.kth_fct_val(i, x)**2
        return val

    def grad(self, x):
        """
        :param x: a point in domain
        :return: function's gradient at x
        """
        grad = numpy.zeros(self.size)
        for i in xrange(1, self.size+1):
            if i % 4 == 1:
                grad[i-1] = 2 * self.kth_fct_val(i, x) + 4 * math.sqrt(10.0)*(x[i-1] - x[i+2])*self.kth_fct_val(i+3, x)
            elif i % 4 == 2:
                grad[i-1] = 20*self.kth_fct_val(i-1, x)+4*self.kth_fct_val(i+1, x)*(x[i-1]-2*x[i])
            elif i % 4 == 3:
                grad[i-1] = 2*math.sqrt(5.0)*self.kth_fct_val(i-1, x)- 8*self.kth_fct_val(i, x)*(x[i-2]-2*x[i-1])
            else:
                grad[i-1] = -2*math.sqrt(5.0)*self.kth_fct_val(i-2, x) - 4*math.sqrt(10.0)*self.kth_fct_val(i, x)*(x[i-4]-x[i-1])

        return grad

    def hessian(self, x):
        """
        :param x: a point in domain
        :return: function's Hessian matrix at x
        """
        H = numpy.zeros(shape =(self.size, self.size))
        for i in xrange(1, self.size+1):
            if i % 4 == 1:
                H[i-1, i-1] = 4*math.sqrt(10)*self.kth_fct_val(i+3, x)+2+80*(x[i-1]-x[i+2])**2
                H[i-1, i+2] = -4*math.sqrt(10)*self.kth_fct_val(i+3, x) - 80*(x[i-1] - x[i+2])**2
                H[i-1, i] = 20.0
            elif i % 4 == 2:
                H[i-1, i-1] = 4*self.kth_fct_val(i+1,x)+200+8*(x[i-1]-2*x[i])**2
                H[i-1, i] = -8*self.kth_fct_val(i+1,x) -16*(x[i-1]-2*x[i])**2
                H[i-1, i-2] = 20.0
            elif i % 4 == 3:
                H[i-1, i-1] = 16*self.kth_fct_val(i,x)+10+32*(x[i-2]-2*x[i-1])**2
                H[i-1, i-2] = -8*self.kth_fct_val(i,x)-16*(x[i-2]-2*x[i-1])**2
                H[i-1, i] = -10.0
            else:
                H[i-1, i-1] = 4*math.sqrt(10)*self.kth_fct_val(i,x)+10+80*(x[i-4]-x[i-1])**2
                H[i-1, i-4] = -4*math.sqrt(10)*self.kth_fct_val(i,x)-80*(x[i-4]-x[i-1])**2
                H[i-1, i-2] = -10.0

        return H
    def get_answer(self):
        return numpy.zeros(self.size)

