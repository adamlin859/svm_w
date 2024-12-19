#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>
#include "svm.h"
#ifdef _OPENMP
#include <omp.h>
#endif

int libsvm_version = LIBSVM_VERSION;
typedef float Qfloat;
typedef signed char schar;
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
static inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for(int t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
static void (*svm_print_string) (const char *) = &print_string_stdout;
#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsnprintf(buf,BUFSIZ,fmt,ap);
	va_end(ap);
	(*svm_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache
{
public:
	Cache(int l,size_t size);
	~Cache();

	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
	int get_data(const int index, Qfloat **data, int len);
	void swap_index(int i, int j);
private:
	int l;
	size_t size;
	struct head_t
	{
		head_t *prev, *next;	// a circular list
		Qfloat *data;
		int len;		// data[0,len) is cached in this entry
	};

	head_t *head;
	head_t lru_head;
	void lru_delete(head_t *h);
	void lru_insert(head_t *h);
};

Cache::Cache(int l_,size_t size_):l(l_),size(size_)
{
	head = (head_t *)calloc(l,sizeof(head_t));	// initialized to 0
	size /= sizeof(Qfloat);
	size_t header_size = l * sizeof(head_t) / sizeof(Qfloat);
	size = max(size, 2 * (size_t) l + header_size) - header_size;  // cache must be large enough for two columns
	lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache()
{
	for(head_t *h = lru_head.next; h != &lru_head; h=h->next)
		free(h->data);
	free(head);
}

void Cache::lru_delete(head_t *h)
{
	// delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void Cache::lru_insert(head_t *h)
{
	// insert to last position
	h->next = &lru_head;
	h->prev = lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

int Cache::get_data(const int index, Qfloat **data, int len)
{
	head_t *h = &head[index];
	if(h->len) lru_delete(h);
	int more = len - h->len;

	if(more > 0)
	{
		// free old space
		while(size < (size_t)more)
		{
			head_t *old = lru_head.next;
			lru_delete(old);
			free(old->data);
			size += old->len;
			old->data = 0;
			old->len = 0;
		}

		// allocate new space
		h->data = (Qfloat *)realloc(h->data,sizeof(Qfloat)*len);
		size -= more;  // previous while loop guarantees size >= more and subtraction of size_t variable will not underflow
		swap(h->len,len);
	}

	lru_insert(h);
	*data = h->data;
	return len;
}

void Cache::swap_index(int i, int j)
{
	if(i==j) return;

	if(head[i].len) lru_delete(&head[i]);
	if(head[j].len) lru_delete(&head[j]);
	swap(head[i].data,head[j].data);
	swap(head[i].len,head[j].len);
	if(head[i].len) lru_insert(&head[i]);
	if(head[j].len) lru_insert(&head[j]);

	if(i>j) swap(i,j);
	for(head_t *h = lru_head.next; h!=&lru_head; h=h->next)
	{
		if(h->len > i)
		{
			if(h->len > j)
				swap(h->data[i],h->data[j]);
			else
			{
				// give up
				lru_delete(h);
				free(h->data);
				size += h->len;
				h->data = 0;
				h->len = 0;
			}
		}
	}
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix {
public:
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const = 0;
	virtual ~QMatrix() {}
};

class Kernel: public QMatrix {
public:
	Kernel(int l, svm_node * const * x, const svm_parameter& param);
	virtual ~Kernel();

	static double k_function(const svm_node *x, const svm_node *y,
				 const svm_parameter& param);
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const	// no so const...
	{
		swap(x[i],x[j]);
		if(x_square) swap(x_square[i],x_square[j]);
	}
protected:

	double (Kernel::*kernel_function)(int i, int j) const;

private:
	const svm_node **x;
	double *x_square;

	// svm_parameter
	const int kernel_type;
	const int degree;
	const double gamma;
	const double coef0;

	static double dot(const svm_node *px, const svm_node *py);
	double kernel_linear(int i, int j) const
	{
		return dot(x[i],x[j]);
	}
	double kernel_poly(int i, int j) const
	{
		return powi(gamma*dot(x[i],x[j])+coef0,degree);
	}
	double kernel_rbf(int i, int j) const
	{
		return exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
	}
	double kernel_sigmoid(int i, int j) const
	{
		return tanh(gamma*dot(x[i],x[j])+coef0);
	}
	double kernel_precomputed(int i, int j) const
	{
		return x[i][(int)(x[j][0].value)].value;
	}
};

Kernel::Kernel(int l, svm_node * const * x_, const svm_parameter& param)
:kernel_type(param.kernel_type), degree(param.degree),
 gamma(param.gamma), coef0(param.coef0)
{
	switch(kernel_type)
	{
		case LINEAR:
			kernel_function = &Kernel::kernel_linear;
			break;
		case POLY:
			kernel_function = &Kernel::kernel_poly;
			break;
		case RBF:
			kernel_function = &Kernel::kernel_rbf;
			break;
		case SIGMOID:
			kernel_function = &Kernel::kernel_sigmoid;
			break;
		case PRECOMPUTED:
			kernel_function = &Kernel::kernel_precomputed;
			break;
	}

	clone(x,x_,l);

	if(kernel_type == RBF)
	{
		x_square = new double[l];
		for(int i=0;i<l;i++)
			x_square[i] = dot(x[i],x[i]);
	}
	else
		x_square = 0;
}

Kernel::~Kernel()
{
	delete[] x;
	delete[] x_square;
}

double Kernel::dot(const svm_node *px, const svm_node *py)
{
	double sum = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}
	}
	return sum;
}

double Kernel::k_function(const svm_node *x, const svm_node *y,
			  const svm_parameter& param)
{
	switch(param.kernel_type)
	{
		case LINEAR:
			return dot(x,y);
		case POLY:
			return powi(param.gamma*dot(x,y)+param.coef0,param.degree);
		case RBF:
		{
			double sum = 0;
			while(x->index != -1 && y->index !=-1)
			{
				if(x->index == y->index)
				{
					double d = x->value - y->value;
					sum += d*d;
					++x;
					++y;
				}
				else
				{
					if(x->index > y->index)
					{
						sum += y->value * y->value;
						++y;
					}
					else
					{
						sum += x->value * x->value;
						++x;
					}
				}
			}

			while(x->index != -1)
			{
				sum += x->value * x->value;
				++x;
			}

			while(y->index != -1)
			{
				sum += y->value * y->value;
				++y;
			}

			return exp(-param.gamma*sum);
		}
		case SIGMOID:
			return tanh(param.gamma*dot(x,y)+param.coef0);
		case PRECOMPUTED:  //x: test (validation), y: SV
			return x[(int)(y->value)].value;
		default:
			return 0;  // Unreachable
	}
}

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver {
public:
	Solver() {};
	virtual ~Solver() {};

	struct SolutionInfo {
		double obj;
		double rho;
		double rho_star;
		double upper_bound_p;
		double upper_bound_n;
		double r;	// for Solver_NU
	};

	void Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking);
	void Solve_w(int l, const QMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking, svm_model *model, const svm_problem& prob);
	void Solve_plus(int l, const QMatrix& Q, const QMatrix& Q_star, const QMatrix& Q_star_beta, const schar *y_,
			     double *alpha_, double *beta_, double Cp, double Cn, double tau_, double eps,
			     SolutionInfo* si, int shrinking);
protected:
	int active_size;
	schar *y;
	double *G;		// gradient of objective function
	double *G_w;	// gradient of objective function \hat{w}
	enum { LOWER_BOUND, UPPER_BOUND, FREE };
	char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
	double *alpha;
	const QMatrix *Q;
	const double *QD;
	double eps;
	double Cp,Cn;
	double *p;
	int *active_set;
	double *G_bar;		// gradient, if we treat free variables as 0
	int l;
	bool unshrink;	// XXX

	// SVM plus
	double *g;
	double *g_beta;
	double *g_init;
	double *g_beta_init;
	char *beta_status;
	double *beta;
	const QMatrix *Q_star;
	const QMatrix *Q_star_beta;
	const double *QD_star;
	const double *QD_star_beta;
	int *active_set_beta;
	int *true_act_set;
	int *true_act_set_beta;
	int active_size_beta;
	double tau;

	double get_C(int i)
	{
		return (y[i] > 0)? Cp : Cn;
	}
	void update_alpha_status(int i)
	{
		if(alpha[i] >= get_C(i))
			alpha_status[i] = UPPER_BOUND;
		else if(alpha[i] <= 0)
			alpha_status[i] = LOWER_BOUND;
		else alpha_status[i] = FREE;
	}

	void update_beta_status(int i)
	{
		if(beta[i] <= 1e-8)
			beta_status[i] = LOWER_BOUND;
		else beta_status[i] = FREE;
	}

	bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
	bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
	bool is_free(int i) { return alpha_status[i] == FREE; }
	void swap_index(int i, int j);
	void reconstruct_gradient();
	void reconstruct_gradient_w();
	void reconstruct_gradient_plus();
	virtual int select_working_set(int &i, int &j);
	virtual double calculate_rho();
	virtual void do_shrinking();

	// SVM plus
	bool is_lower_bound_beta(int i) { return beta_status[i] == LOWER_BOUND; }
	bool is_free_beta(int i) { return beta_status[i] == FREE; }
	void swap_index_beta(int i, int j);
	void swap_index_alpha(int i, int j);
	virtual int select_working_set_plus(int& set_type, int& i, int& j, int& k, int iter);
	virtual void calculate_rho_plus(double& rho, double& rho_star);
	virtual void do_shrinking_plus();
private:
	bool be_shrunk(int i, double Gmax1, double Gmax2);
	bool be_shrunk_alpha(int i, double max_B1, double max_A1, double max_A2,  double min_B1B2, double min_A1A3, double min_A2A4);
	bool be_shrunk_beta(int i, double max_B1, double max_A1, double max_A2,  double min_B1B2, double min_A1A3, double min_A2A4);
 
};

void Solver::swap_index(int i, int j)
{
	Q->swap_index(i,j);
	swap(y[i],y[j]);
	swap(G[i],G[j]);
	swap(alpha_status[i],alpha_status[j]);
	swap(alpha[i],alpha[j]);
	swap(p[i],p[j]);
	swap(active_set[i],active_set[j]);
	swap(G_bar[i],G_bar[j]);
}

void Solver::swap_index_alpha(int i, int j)
{
  Q->swap_index(i,j);
  Q_star->swap_index(i,j);
  swap(y[i],y[j]);
  swap(alpha_status[i],alpha_status[j]);
  swap(alpha[i],alpha[j]);
  swap(true_act_set[active_set[i]],true_act_set[active_set[j]]);
  swap(active_set[i],active_set[j]);
  swap(G[i],G[j]);
  swap(g[i],g[j]);
  swap(g_init[i],g_init[j]);
}

void Solver::swap_index_beta(int i, int j)
{
  Q_star_beta->swap_index(i,j);
  swap(beta_status[i],beta_status[j]);
  swap(beta[i],beta[j]);
  swap(true_act_set_beta[active_set_beta[i]],true_act_set_beta[active_set_beta[j]]);
  swap(active_set_beta[i],active_set_beta[j]);
  swap(g_beta[i],g_beta[j]);
  swap(g_beta_init[i],g_beta_init[j]);
}

void Solver::reconstruct_gradient()
{
	// reconstruct inactive elements of G from G_bar and free variables

	if(active_size == l) return;

	int i,j;
	int nr_free = 0;

	for(j=active_size;j<l;j++)
		G[j] = G_bar[j] + p[j];

	for(j=0;j<active_size;j++)
		if(is_free(j))
			nr_free++;

	if(2*nr_free < active_size)
		info("\nWARNING: using -h 0 may be faster\n");

	if (nr_free*l > 2*active_size*(l-active_size))
	{
		for(i=active_size;i<l;i++)
		{
			const Qfloat *Q_i = Q->get_Q(i,active_size);
			for(j=0;j<active_size;j++)
				if(is_free(j))
					G[i] += alpha[j] * Q_i[j];
		}
	}
	else
	{
		for(i=0;i<active_size;i++)
			if(is_free(i))
			{
				const Qfloat *Q_i = Q->get_Q(i,l);
				double alpha_i = alpha[i];
				for(j=active_size;j<l;j++)
					G[j] += alpha_i * Q_i[j];
			}
	}
}

void Solver::reconstruct_gradient_w()
{
	// reconstruct inactive elements of G from G_bar and free variables

	if(active_size == l) return;

	int i,j;
	int nr_free = 0;

	for(j=active_size;j<l;j++)
		G[j] = G_bar[j] + p[j];

	for(j=0;j<active_size;j++)
		if(is_free(j))
			nr_free++;

	if(2*nr_free < active_size)
		info("\nWARNING: using -h 0 may be faster\n");

	if (nr_free*l > 2*active_size*(l-active_size))
	{
		for(i=active_size;i<l;i++)
		{
			const Qfloat *Q_i = Q->get_Q(i,active_size);
			for(j=0;j<active_size;j++)
				if(is_free(j))
					G[i] += alpha[j] * Q_i[j];
			
			G[i] += G_w[i];
		}
		
	}
	else
	{
		for(i=0;i<active_size;i++)
			if(is_free(i))
			{
				const Qfloat *Q_i = Q->get_Q(i,l);
				double alpha_i = alpha[i];
				for(j=active_size;j<l;j++)
					G[j] += alpha_i * Q_i[j];

				G[j] += G_w[j];
			}
	}
}

void Solver::reconstruct_gradient_plus()
{
	int i, j, true_i, act_set_i;

	if(active_size < l) 
	{
		for(i=active_size;i<l;i++) {
			const Qfloat *Q_i = Q->get_Q(i,l);
			const Qfloat *Q_i_star = Q_star->get_Q(i,l);

			true_i = active_set[i];
			act_set_i = true_act_set_beta[true_i];

			const Qfloat *Q_i_star_beta = Q_star_beta->get_Q(act_set_i,l);
			G[i] = 0;
			g[i] = g_init[i];
			for(j=0;j<l;j++)
			if(alpha[j]>1e-8) 
			{
				G[i] += alpha[j] * y[j] * Q_i[j];
				g[i] += alpha[j] * Q_i_star[j];
			}
			for(j=0;j<l;j++)
				if(beta[j]>1e-8) 
					g[i] += beta[j] * Q_i_star_beta[j];
		}
	}

	if(active_size_beta < l) 
	{
		for(i=active_size_beta;i<l;i++) 
		{
			const Qfloat *Q_i_star_beta = Q_star_beta->get_Q(i,l);

			true_i = active_set_beta[i];
			act_set_i = true_act_set[true_i];
			const Qfloat *Q_i_star = Q_star->get_Q(act_set_i,l);

			g_beta[i] = g_beta_init[i];

			for(j=0;j<l;j++)
				if(beta[j]>1e-8) 
					g_beta[i] += beta[j] * Q_i_star_beta[j];

			for(j=0;j<l;j++)
				if(alpha[j]>1e-8) 
					g_beta[i] += alpha[j] * Q_i_star[j];
		}
	}
}

void Solver::Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking)
{
	this->l = l;
	this->Q = &Q;
	QD=Q.get_QD();
	clone(p, p_,l);
	clone(y, y_,l);
	clone(alpha,alpha_,l);
	this->Cp = Cp;
	this->Cn = Cn;
	this->eps = eps;
	unshrink = false;

	// initialize alpha_status
	{
		alpha_status = new char[l];
		for(int i=0;i<l;i++)
			update_alpha_status(i);
	}

	// initialize active set (for shrinking)
	{
		active_set = new int[l];
		for(int i=0;i<l;i++)
			active_set[i] = i;
		active_size = l;
	}

	// initialize gradient
	{
		G = new double[l];
		G_bar = new double[l];
		int i;
		for(i=0;i<l;i++)
		{
			G[i] = p[i];
			G_bar[i] = 0;
		}
		for(i=0;i<l;i++)
			if(!is_lower_bound(i))
			{
				const Qfloat *Q_i = Q.get_Q(i,l);
				double alpha_i = alpha[i];
				int j;
				for(j=0;j<l;j++)
					G[j] += alpha_i*Q_i[j];
				if(is_upper_bound(i))
					for(j=0;j<l;j++)
						G_bar[j] += get_C(i) * Q_i[j];
					
			}
	}

	// optimization step

	int iter = 0;
	int max_iter = max(10000000, l>INT_MAX/100 ? INT_MAX : 100*l);
	int counter = min(l,1000)+1;

	while(iter < max_iter)
	{
		// show progress and do shrinking

		if(--counter == 0)
		{
			counter = min(l,1000);
			if(shrinking) do_shrinking();
			info(".");
		}

		int i,j;
		if(select_working_set(i,j)!=0)
		{
			// reconstruct the whole gradient
			reconstruct_gradient();
			// reset active set size and check
			active_size = l;
			info("*");
			if(select_working_set(i,j)!=0)
				break;
			else
				counter = 1;	// do shrinking next iteration
		}

		++iter;

		// update alpha[i] and alpha[j], handle bounds carefully

		const Qfloat *Q_i = Q.get_Q(i,active_size);
		const Qfloat *Q_j = Q.get_Q(j,active_size);

		double C_i = get_C(i);
		double C_j = get_C(j);

		double old_alpha_i = alpha[i];
		double old_alpha_j = alpha[j];

		if(y[i]!=y[j])
		{
			double quad_coef = QD[i]+QD[j]+2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (-G[i]-G[j])/quad_coef;
			double diff = alpha[i] - alpha[j];
			alpha[i] += delta;
			alpha[j] += delta;

			if(diff > 0)
			{
				if(alpha[j] < 0)
				{
					alpha[j] = 0;
					alpha[i] = diff;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = -diff;
				}
			}
			if(diff > C_i - C_j)
			{
				if(alpha[i] > C_i)
				{
					alpha[i] = C_i;
					alpha[j] = C_i - diff;
				}
			}
			else
			{
				if(alpha[j] > C_j)
				{
					alpha[j] = C_j;
					alpha[i] = C_j + diff;
				}
			}
		}
		else
		{
			double quad_coef = QD[i]+QD[j]-2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (G[i]-G[j])/quad_coef;
			double sum = alpha[i] + alpha[j];
			alpha[i] -= delta;
			alpha[j] += delta;

			if(sum > C_i)
			{
				if(alpha[i] > C_i)
				{
					alpha[i] = C_i;
					alpha[j] = sum - C_i;
				}
			}
			else
			{
				if(alpha[j] < 0)
				{
					alpha[j] = 0;
					alpha[i] = sum;
				}
			}
			if(sum > C_j)
			{
				if(alpha[j] > C_j)
				{
					alpha[j] = C_j;
					alpha[i] = sum - C_j;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = sum;
				}
			}
		}

		// update G

		double delta_alpha_i = alpha[i] - old_alpha_i;
		double delta_alpha_j = alpha[j] - old_alpha_j;

		for(int k=0;k<active_size;k++)
		{
			G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
		}

		// update alpha_status and G_bar

		{
			bool ui = is_upper_bound(i);
			bool uj = is_upper_bound(j);
			update_alpha_status(i);
			update_alpha_status(j);
			int k;
			if(ui != is_upper_bound(i))
			{
				Q_i = Q.get_Q(i,l);
				if(ui)
					for(k=0;k<l;k++)
						G_bar[k] -= C_i * Q_i[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_i * Q_i[k];
			}

			if(uj != is_upper_bound(j))
			{
				Q_j = Q.get_Q(j,l);
				if(uj)
					for(k=0;k<l;k++)
						G_bar[k] -= C_j * Q_j[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_j * Q_j[k];
			}
		}
	}

	if(iter >= max_iter)
	{
		if(active_size < l)
		{
			// reconstruct the whole gradient to calculate objective value
			reconstruct_gradient();
			active_size = l;
			info("*");
		}
		fprintf(stderr,"\nWARNING: reaching max number of iterations\n");
	}

	// calculate rho

	si->rho = calculate_rho();

	// calculate objective value
	{
		double v = 0;
		int i;
		for(i=0;i<l;i++)
			v += alpha[i] * (G[i] + p[i]);

		si->obj = v/2;
	}

	// put back the solution
	{
		for(int i=0;i<l;i++)
			alpha_[active_set[i]] = alpha[i];
	}

	// juggle everything back
	/*{
		for(int i=0;i<l;i++)
			while(active_set[i] != i)
				swap_index(i,active_set[i]);
				// or Q.swap_index(i,active_set[i]);
	}*/

	si->upper_bound_p = Cp;
	si->upper_bound_n = Cn;

	info("\noptimization finished, #iter = %d\n",iter);

	delete[] p;
	delete[] y;
	delete[] alpha;
	delete[] alpha_status;
	delete[] active_set;
	delete[] G;
	delete[] G_bar;
}

void Solver::Solve_w(int l, const QMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking, svm_model *model, const svm_problem& prob)
{

	this->l = l;
	this->Q = &Q;
	QD=Q.get_QD();
	clone(p, p_,l);
	clone(y, y_,l);
	clone(alpha,alpha_,l);
	this->Cp = Cp;
	this->Cn = Cn;
	this->eps = eps;
	unshrink = false;

	// initialize alpha_status
	{
		alpha_status = new char[l];
		for(int i=0;i<l;i++)
			update_alpha_status(i);
	}

	// initialize active set (for shrinking)
	{
		active_set = new int[l];
		for(int i=0;i<l;i++)
			active_set[i] = i;
		active_size = l;
	}

	// initialize gradient
	{
		G = new double[l];
		G_w = new double[l];
		G_bar = new double[l];
		int i;
		for(i=0;i<l;i++)
		{
			G[i] = p[i];
			G_bar[i] = 0;
		}
		for(i=0;i<l;i++)
			if(!is_lower_bound(i))
			{
				const Qfloat *Q_i = Q.get_Q(i,l);
				double alpha_i = alpha[i];
				int j;
				for(j=0;j<l;j++)
					G[j] += alpha_i*Q_i[j];
				if(is_upper_bound(i))
					for(j=0;j<l;j++)
						G_bar[j] += get_C(i) * Q_i[j];
			}

		
		
		for (i = 0; i < l; i++) 
		{
			double *sv_coef = model->sv_coef[0];
			double sum = 0;
			int j;

#ifdef _OPENMP
#pragma omp parallel for private(j) reduction(+:sum) schedule(guided)
#endif
			for(j=0;j<model->l;j++)
			{
				sum += y[i] * sv_coef[j] * Kernel::k_function(prob.x[i],model->SV[j],model->param);
			}
			// print G[i] and sum
			G_w[i] = sum;
			G[i] += G_w[i]; 
		}
	}

	// optimization step

	int iter = 0;
	int max_iter = max(10000000, l>INT_MAX/100 ? INT_MAX : 100*l);
	int counter = min(l,1000)+1;

	while(iter < max_iter)
	{
		// show progress and do shrinking

		if(--counter == 0)
		{
			counter = min(l,1000);
			if(shrinking) do_shrinking();
			info(".");
		}

		int i,j;
		if(select_working_set(i,j)!=0)
		{
			// reconstruct the whole gradient
			reconstruct_gradient_w();
			// reset active set size and check
			active_size = l;
			info("*");
			if(select_working_set(i,j)!=0)
				break;
			else
				counter = 1;	// do shrinking next iteration
		}

		++iter;

		// update alpha[i] and alpha[j], handle bounds carefully

		const Qfloat *Q_i = Q.get_Q(i,active_size);
		const Qfloat *Q_j = Q.get_Q(j,active_size);

		double C_i = get_C(i);
		double C_j = get_C(j);

		double old_alpha_i = alpha[i];
		double old_alpha_j = alpha[j];

		if(y[i]!=y[j])
		{
			double quad_coef = QD[i]+QD[j]+2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (-G[i]-G[j])/quad_coef;
			double diff = alpha[i] - alpha[j];
			alpha[i] += delta;
			alpha[j] += delta;

			if(diff > 0)
			{
				if(alpha[j] < 0)
				{
					alpha[j] = 0;
					alpha[i] = diff;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = -diff;
				}
			}
			if(diff > C_i - C_j)
			{
				if(alpha[i] > C_i)
				{
					alpha[i] = C_i;
					alpha[j] = C_i - diff;
				}
			}
			else
			{
				if(alpha[j] > C_j)
				{
					alpha[j] = C_j;
					alpha[i] = C_j + diff;
				}
			}
		}
		else
		{
			double quad_coef = QD[i]+QD[j]-2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (G[i]-G[j])/quad_coef;
			double sum = alpha[i] + alpha[j];
			alpha[i] -= delta;
			alpha[j] += delta;

			if(sum > C_i)
			{
				if(alpha[i] > C_i)
				{
					alpha[i] = C_i;
					alpha[j] = sum - C_i;
				}
			}
			else
			{
				if(alpha[j] < 0)
				{
					alpha[j] = 0;
					alpha[i] = sum;
				}
			}
			if(sum > C_j)
			{
				if(alpha[j] > C_j)
				{
					alpha[j] = C_j;
					alpha[i] = sum - C_j;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = sum;
				}
			}
		}

		// update G

		double delta_alpha_i = alpha[i] - old_alpha_i;
		double delta_alpha_j = alpha[j] - old_alpha_j;

		for(int k=0;k<active_size;k++)
		{
			G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
		}

		// update alpha_status and G_bar

		{
			bool ui = is_upper_bound(i);
			bool uj = is_upper_bound(j);
			update_alpha_status(i);
			update_alpha_status(j);
			int k;
			if(ui != is_upper_bound(i))
			{
				Q_i = Q.get_Q(i,l);
				if(ui)
					for(k=0;k<l;k++)
						G_bar[k] -= C_i * Q_i[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_i * Q_i[k];
			}

			if(uj != is_upper_bound(j))
			{
				Q_j = Q.get_Q(j,l);
				if(uj)
					for(k=0;k<l;k++)
					{
						G_bar[k] -= C_j * Q_j[k];
						G_bar[k] += G_w[k];
					}
				else
					for(k=0;k<l;k++)
					{
						G_bar[k] += C_j * Q_j[k];
						G_bar[k] += G_w[k];
					}
				
			}
		}
	}

	if(iter >= max_iter)
	{
		if(active_size < l)
		{
			// reconstruct the whole gradient to calculate objective value
			reconstruct_gradient_w();
			active_size = l;
			info("*");
		}
		fprintf(stderr,"\nWARNING: reaching max number of iterations\n");
	}

	// calculate rho

	si->rho = calculate_rho();

	// calculate objective value
	{
		double v = 0;
		int i;
		for(i=0;i<l;i++)
			v += alpha[i] * (G[i] + p[i]);

		si->obj = v/2;
	}

	// put back the solution
	{
		for(int i=0;i<l;i++)
			alpha_[active_set[i]] = alpha[i];
	}

	// juggle everything back
	/*{
		for(int i=0;i<l;i++)
			while(active_set[i] != i)
				swap_index(i,active_set[i]);
				// or Q.swap_index(i,active_set[i]);
	}*/

	si->upper_bound_p = Cp;
	si->upper_bound_n = Cn;

	info("\noptimization finished, #iter = %d\n",iter);

	delete[] p;
	delete[] y;
	delete[] alpha;
	delete[] alpha_status;
	delete[] active_set;
	delete[] G;
	delete[] G_w;
	delete[] G_bar;
}

void Solver::Solve_plus(int l, const QMatrix& Q, const QMatrix& Q_star, const QMatrix& Q_star_beta, const schar *y_,
			     double *alpha_, double *beta_, double Cp, double Cn, double tau_, double eps,
			     SolutionInfo* si, int shrinking)
{
	int i,j;
	this->l = l;
	this->Q = &Q;
	this->Q_star = &Q_star;
	this->Q_star_beta = &Q_star_beta;
	QD = Q.get_QD();
	QD_star = Q_star.get_QD();
	QD_star_beta = Q_star_beta.get_QD();	
	clone(y, y_,l);
	clone(alpha,alpha_,l);
	clone(beta,beta_,l);
	this->Cp = Cp;
	this->Cn = Cn;
	this->eps = eps;
	tau = tau_;
	unshrink = false;

	{
		alpha_status = new char[l];
		for(i=0;i<l;i++)
			update_alpha_status(i);
	}

	{
		beta_status = new char[l];
		for(i=0;i<l;i++)
			update_beta_status(i);
	}

	{
		// initialize gradient
		G = new double[l];
		g = new double[l];
		g_beta = new double[l];
		g_init = new double[l];
		g_beta_init = new double[l];

		for(i=0; i<l; i++) 
		{
			G[i] = 0;
			g[i] = 0;
			g_init[i] =0;
		}

		for(i=0;i<l;i++)
		{
			const Qfloat *Q_i_star = Q_star.get_Q(i,l);
			for(j=0; j<l; j++) 
			{
				g[j] -= Cp*Q_i_star[j];
				g_init[j] -= Cp*Q_i_star[j];
			}

			if(!is_lower_bound(i))
			{
				const Qfloat *Q_i = Q.get_Q(i,l);
				double alpha_i = alpha[i];
				double y_i = y[i];
				for(j=0;j<l;j++)
				{
					G[j] += alpha_i*y_i*Q_i[j];
					g[j] += alpha_i*Q_i_star[j];
				}
			}
			if(!is_lower_bound_beta(i)) 
			{
				double beta_i = beta[i];
				for(j=0;j<l;j++) 
					g[j] += beta_i*Q_i_star[j];    
			}
		}

		for(i=0; i<l; i++) {
			g_beta[i] = g[i];
			g_beta_init[i] = g_init[i];
		}			
	}

	active_set = new int[l];
	active_set_beta = new int[l];
	true_act_set = new int[l];
	true_act_set_beta = new int[l];

	for(int i=0; i<l; i++) 
	{
		active_set[i] = i;
		active_set_beta[i] = i;
		true_act_set[i] = i;
		true_act_set_beta[i] = i;
	}
	active_size = l;
	active_size_beta = l;

	int counter = min(l,1000)+1;

	// optimization step
	int iter = 0, y_i, y_j;
	Qfloat *Q_i, *Q_j, *Q_i_star, *Q_j_star, *Q_k_star, *Q_i_star_beta, *Q_j_star_beta, *Q_k_star_beta;
	double Delta, beta_i_old, beta_j_old, alpha_i_old, alpha_j_old, beta_k_old, nominator, denominator, min_alpha, alpha_change;
	double diff_i, diff_j, diff_k, beta_i, beta_k, alpha_i, diff_i_y, diff_j_y;
	int true_i, true_j, true_k, act_set_i, act_set_j, act_set_k;

	while(iter<1e7) 
	{
		int i,j,k,set_type,r;

		if(--counter == 0) 
		{
			counter = min(l,1000);
			if(shrinking) 
				do_shrinking_plus();
		}

		if(select_working_set_plus(set_type, i,j,k, iter) != 0) 
		{
			// reconstruct the whole gradient
			reconstruct_gradient_plus();
			// reset active set size and check
			active_size = l;
			active_size_beta = l;
			
			if(select_working_set_plus(set_type, i,j,k, iter) != 0)
				break;
			else
				counter = 1;	// do shrinking next iteration
		}

		++iter;

		switch(set_type)
		{
			case BETA_I_BETA_J: 
				Q_i_star_beta = Q_star_beta.get_Q(i,active_size_beta);
				Q_j_star_beta = Q_star_beta.get_Q(j,active_size_beta);
				beta_i_old = beta[i];
				beta_j_old = beta[j];
				Delta = beta_i_old + beta_j_old;
				beta[i] += (g_beta[j]-g_beta[i])/(Q_i_star_beta[i]+Q_j_star_beta[j]-2*Q_i_star_beta[j]);
				beta_i = beta[i];
				if (beta_i < 0)
					beta[i] = 0;
				if (beta_i > Delta)
					beta[i] = Delta;
					beta[j] = Delta - beta[i];

				diff_i = beta[i]-beta_i_old;
				diff_j = beta[j]-beta_j_old;
				for(r=0; r<active_size_beta; r++) 
					g_beta[r] += diff_i*Q_i_star_beta[r]+diff_j*Q_j_star_beta[r]; 
				
				update_beta_status(i);
				update_beta_status(j);
				break;

			case ALPHA_I_ALPHA_J:
				Q_i = Q.get_Q(i,active_size);
				Q_j = Q.get_Q(j,active_size);
				Q_i_star = Q_star.get_Q(i,active_size);
				Q_j_star = Q_star.get_Q(j,active_size);
				alpha_i_old = alpha[i];
				alpha_j_old = alpha[j];
				y_i = y[i];
				y_j = y[j];
				Delta = alpha_i_old + alpha_j_old;
				nominator = y_j*G[j]-y_i*G[i]+(g[j]-g[i])/tau;
				denominator = Q_i[i]+Q_j[j]-2*Q_i[j]+(Q_i_star[i]+Q_j_star[j]-2*Q_i_star[j])/tau;
				alpha[i] += nominator/denominator;
				alpha_i = alpha[i];
				if (alpha_i < 0)
					alpha[i] = 0;
				if (alpha_i > Delta)
					alpha[i] = Delta;
				alpha[j] = Delta - alpha[i];

				diff_i = alpha[i]-alpha_i_old;
				diff_j = alpha[j]-alpha_j_old;
				diff_i_y = diff_i * y_i;
				diff_j_y = diff_j * y_j;      
				for (r=0; r<active_size; r++) 
				{
					G[r] += diff_i_y*Q_i[r]+diff_j_y*Q_j[r];
					g[r] += diff_i*Q_i_star[r]+diff_j*Q_j_star[r]; 
				}

				true_i = active_set[i];
				act_set_i = true_act_set_beta[true_i];
				true_j = active_set[j];
				act_set_j = true_act_set_beta[true_j];
				Q_i_star_beta = Q_star_beta.get_Q(act_set_i,active_size_beta);
				Q_j_star_beta = Q_star_beta.get_Q(act_set_j,active_size_beta);

				for (r=0; r<active_size_beta; r++) 
					g_beta[r] += diff_i*Q_i_star_beta[r]+diff_j*Q_j_star_beta[r]; 

				update_alpha_status(i);
				update_alpha_status(j);
				break;

			case ALPHA_I_ALPHA_J_BETA_K:
				Q_i = Q.get_Q(i,active_size);
				Q_j = Q.get_Q(j,active_size);
				Q_i_star = Q_star.get_Q(i,active_size);
				Q_j_star = Q_star.get_Q(j,active_size);
				Q_k_star_beta = Q_star_beta.get_Q(k,active_size_beta);

				true_k = active_set_beta[k];
				act_set_k = true_act_set[true_k];
				Q_k_star = Q_star.get_Q(act_set_k, active_size);

				alpha_i_old = alpha[i];
				alpha_j_old = alpha[j];
				beta_k_old = beta[k];
				y_i = y[i];
				y_j = y[j];
				if(alpha_i_old < alpha_j_old)
					min_alpha = alpha_i_old;
				else
					min_alpha = alpha_j_old;

				Delta = beta_k_old + 2*min_alpha;
				nominator = y[i]*G[i]+y[j]*G[j]-2+(g[i]+g[j]-2*g_beta[k])/tau;
				denominator = Q_i[i]+Q_j[j]-2*Q_i[j]+(Q_i_star[i]+Q_j_star[j]+2*Q_i_star[j]-4*Q_k_star[i]-4*Q_k_star[j]+4*Q_k_star_beta[k])/tau;
				beta[k] += 2*nominator/denominator;
				beta_k = beta[k];
				if (beta_k < 0)
					beta[k] = 0;
				if (beta_k > Delta)
					beta[k] = Delta;
				alpha_change = (beta_k_old-beta[k])/2;
				alpha[i] += alpha_change;
				alpha[j] += alpha_change;

				diff_i = alpha[i]-alpha_i_old;
				diff_j = alpha[j]-alpha_j_old;
				diff_k = beta[k]-beta_k_old;
				diff_i_y = diff_i * y_i;
				diff_j_y = diff_j * y_j;

				for (r=0; r<active_size; r++) 
				{
					G[r] += diff_i_y*Q_i[r]+diff_j_y*Q_j[r];
					g[r] += diff_i*Q_i_star[r]+diff_j*Q_j_star[r]+diff_k*Q_k_star[r]; 
				}

				true_i = active_set[i];
				act_set_i = true_act_set_beta[true_i];
				true_j = active_set[j];
				act_set_j = true_act_set_beta[true_j];
				Q_i_star_beta = Q_star_beta.get_Q(act_set_i,active_size_beta);
				Q_j_star_beta = Q_star_beta.get_Q(act_set_j,active_size_beta);

				for(r=0; r<active_size_beta; r++)
					g_beta[r] += diff_i*Q_i_star_beta[r]+diff_j*Q_j_star_beta[r]+diff_k*Q_k_star_beta[r];

				update_alpha_status(i);
				update_alpha_status(j);
				update_beta_status(k);
					
				break;
		}
	}

	calculate_rho_plus(si->rho,si->rho_star);

	// put back the solution
	for(i=0;i<l;i++) {
		alpha_[active_set[i]] = alpha[i];
		beta_[active_set_beta[i]] = beta[i];
	}
		
	si->upper_bound_p = Cp;
	si->upper_bound_n = Cn;

	info("\noptimization finished, #iter = %d\n",iter);

	si->rho *= -1;

	delete[] G;
	delete[] g;
	delete[] g_init;
	delete[] g_beta;
	delete[] g_beta_init;
	delete[] alpha_status;
	delete[] beta_status;
	delete[] alpha;
	delete[] beta;


}

// return 1 if already optimal, return 0 otherwise
int Solver::select_working_set(int &out_i, int &out_j)
{
	
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	double Gmax = -INF;
	double Gmax2 = -INF;
	int Gmax_idx = -1;
	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for(int t=0;t<active_size;t++)
		if(y[t]==+1)
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmax)
				{
					Gmax = -G[t];
					Gmax_idx = t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmax)
				{
					Gmax = G[t];
					Gmax_idx = t;
				}
		}

	int i = Gmax_idx;
	const Qfloat *Q_i = NULL;
	if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
		Q_i = Q->get_Q(i,active_size);

	for(int j=0;j<active_size;j++)
	{
		if(y[j]==+1)
		{
			if (!is_lower_bound(j))
			{
				double grad_diff=Gmax+G[j];
				if (G[j] >= Gmax2)
					Gmax2 = G[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = QD[i]+QD[j]-2.0*y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				double grad_diff= Gmax-G[j];
				if (-G[j] >= Gmax2)
					Gmax2 = -G[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = QD[i]+QD[j]+2.0*y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	if(Gmax+Gmax2 < eps || Gmin_idx == -1)
		return 1;

	out_i = Gmax_idx;
	out_j = Gmin_idx;
	return 0;
}

// return 1 if already optimal, return 0 otherwise
int Solver::select_working_set_plus(int& set_type, int& i_out, int& j_out, int& k_out, int iter)
{
  double gap[3];

  for(int i=0; i <3; i++)
    gap[i] = -1;

  int i, j, best_B1=-1, best_B1B2=-1, best_A2=-1, best_A2A4=-1, best_A1=-1, best_A1A3=-1, i_ind, j_ind, k_ind;
  int type_selected[3], selected_indices[3][3];
  double max_B1=-1e20, min_B1B2=1e20, max_A2=-1e20, min_A2A4=1e20, max_A1=-1e20, min_A1A3=1e20;
  double alpha_i, g_i, g_j, y_i, y_j, deriv_alpha_i, first_order_criterion;
  double max_z[3], z, absolute_best_z, nominator, nominator_base, denominator_base, j_deriv, tau2, Q_star_ii;
  int best_z_index[3], true_k, act_set_k;
  Qfloat *Q_i, *Q_i_star, *Q_k_star, *Q_k_star_beta;
  double deriv_alpha_i1, deriv_alpha_i2, nominator1, nominator2, nominator_base1, nominator_base2, j_deriv1, j_deriv2;  
  double max_A2_1, max_A2_2, min_A2A4_1, min_A2A4_2, max_A1_1, max_A1_2, min_A1A3_1, min_A1A3_2;

  // first-order working set selection 

  // compute all maxima and minima related to alphas
  for(i=0; i<active_size; i++) {
    alpha_i = alpha[i];
    g_i = g[i];
    y_i = y[i];
    deriv_alpha_i1 = y_i*G[i];
    deriv_alpha_i2 = g_i;
    deriv_alpha_i = deriv_alpha_i1+deriv_alpha_i2/tau;

    // max A2
    if(alpha_i>1e-8 && y_i==-1 && deriv_alpha_i>max_A2) {
      max_A2 = deriv_alpha_i;
      best_A2 = i;
      max_A2_1 = deriv_alpha_i1;
      max_A2_2 = deriv_alpha_i2;      
    }

    // min A2A4
    if(y_i==-1 && deriv_alpha_i<min_A2A4) {
      min_A2A4 = deriv_alpha_i;
      best_A2A4 = i;
      min_A2A4_1 = deriv_alpha_i1;
      min_A2A4_2 = deriv_alpha_i2;      
    }
    
    // max A1
    if(alpha_i>1e-8 && y_i==1 && deriv_alpha_i>max_A1) {
      max_A1 = deriv_alpha_i;
      best_A1 = i;
      max_A1_1 = deriv_alpha_i1;
      max_A1_2 = deriv_alpha_i2;      
    }
    
    // min A1A3
    if(y_i==1 && deriv_alpha_i<min_A1A3) {
      min_A1A3 = deriv_alpha_i;
      best_A1A3 = i;
      min_A1A3_1 = deriv_alpha_i1;
      min_A1A3_2 = deriv_alpha_i2;      
    } 
  }

  // compute all maxima and minima related to betas
  for(i=0; i<active_size_beta; i++) {
    g_i = g_beta[i];

    // max B1
    if(beta[i]>1e-8 && g_i>max_B1) {
      max_B1 = g_i;
      best_B1 = i;
    }

    // min B1B2
    if(g_i<min_B1B2) {
      min_B1B2 = g_i;
      best_B1B2 = i;
    }
  }
 
  max_B1 /= tau;
  min_B1B2 /= tau;

  // select maximal violating pairs
  if(max_B1-min_B1B2 < eps)
    type_selected[0] = 0;
  else {
    type_selected[0] = 1;
    selected_indices[0][0] = best_B1;
    selected_indices[0][1] = best_B1B2;
    gap[0] = max_B1-min_B1B2;
  }

 if(((max_A2 - min_A2A4 < eps) || ((max_A2_1 - min_A2A4_1 < eps) && (max_A2_2 - min_A2A4_2 < eps))) &&
     ((max_A1 - min_A1A3 < eps) || ((max_A1_1 - min_A1A3_1 < eps) && (max_A1_2 - min_A1A3_2 < eps))))
    type_selected[1] = 0;
  else {
    if((max_A2 - min_A2A4 > max_A1 - min_A1A3) && ((max_A2_1 - min_A2A4_1 >= eps) || (max_A2_2 - min_A2A4_2 >= eps))) {
      type_selected[1] = 1;
      selected_indices[1][0] = best_A2;
      selected_indices[1][1] = best_A2A4;
    }
    else {
      if((max_A2 - min_A2A4 <= max_A1 - min_A1A3) && ((max_A1_1 - min_A1A3_1 >= eps) || (max_A1_2 - min_A1A3_2 >= eps))) {
        type_selected[1] = 1;
        selected_indices[1][0] = best_A1;
        selected_indices[1][1] = best_A1A3;
      }
      else
        type_selected[1] = 0;
    }
  }

   if(((2*max_B1+2-min_A1A3-min_A2A4 < eps) || ((2-min_A1A3_1-min_A2A4_1<eps) && (2*max_B1*tau-min_A1A3_2-min_A2A4_2<eps))) &&
     ((max_A1+max_A2-2*min_B1B2-2 < eps) || ((max_A1_1+max_A2_1-2<eps) && (max_A1_2+max_A2_2-2*min_B1B2*tau<eps))))
    type_selected[2] = 0;
  else {
    if((2*max_B1+2-min_A1A3-min_A2A4 > max_A1+max_A2-2*min_B1B2-2) && ((2-min_A1A3_1-min_A2A4_1>=eps) || (2*max_B1*tau-min_A1A3_2-min_A2A4_2>=eps))) {
      type_selected[2] = 1;
      selected_indices[2][0] = best_A1A3;
      selected_indices[2][1] = best_A2A4;
      selected_indices[2][2] = best_B1;
    }
    else {
      if((2*max_B1+2-min_A1A3-min_A2A4 <= max_A1+max_A2-2*min_B1B2-2) && ((max_A1_1+max_A2_1-2>=eps) || (max_A1_2+max_A2_2-2*min_B1B2*tau>=eps))) {
        type_selected[2] = 1;
        selected_indices[2][0] = best_A1;
        selected_indices[2][1] = best_A2;
        selected_indices[2][2] = best_B1B2;
      }
      else
        type_selected[2] = 0;
    }
  }

  if(type_selected[0]+type_selected[1]+type_selected[2] == 0) 
    return 1;

  for(i=0; i<3; i++)
    max_z[i] = -1e20;    

  // second-order working set selection
  if(type_selected[0] == 1) {
     i_ind = selected_indices[0][0];
     g_i = g_beta[i_ind];
     Q_i_star = Q_star_beta->get_Q(i_ind,active_size_beta);
     Q_star_ii = Q_i_star[i_ind];
     tau2 = 2*tau;
     for(j = 0; j < active_size_beta; j++) {
        g_j = g_beta[j];
	if(eps+g_j/tau < g_i/tau) {
	  nominator = g_i-g_j;
	  z = nominator*nominator/(tau2*(Q_star_ii+QD_star_beta[j]-2*Q_i_star[j]));
	  if(z > max_z[0]) {
	    max_z[0] = z;
	    best_z_index[0] = j;
	  }
	}
      }
  }

  if(type_selected[1] == 1) {
     i_ind = selected_indices[1][0];
     y_i = y[i_ind];
     Q_i = Q->get_Q(i_ind,active_size);
     Q_i_star = Q_star->get_Q(i_ind,active_size);
     nominator_base = y_i*G[i_ind] + g[i_ind]/tau;
      nominator_base1 = y_i*G[i_ind];
     nominator_base2 = g[i_ind];
     denominator_base = 2*(Q_i[i_ind]+Q_i_star[i_ind]/tau);
   
     for(j = 0; j < active_size; j++) { 
        y_j = y[j];
        j_deriv = y_j*G[j]+g[j]/tau;
         j_deriv1 = y_j*G[j];
        j_deriv2 = g[j];

        if(y_j == y_i && j_deriv+eps < nominator_base && ((j_deriv1+eps < nominator_base1) || (j_deriv2+eps < nominator_base2))) {
	  nominator = nominator_base-j_deriv;
	  z = nominator*nominator/(denominator_base + 2*(QD[j]-2*Q_i[j]+(QD_star[j]-2*Q_i_star[j])/tau));
	  if(z > max_z[1]) {
	    max_z[1] = z;
	    best_z_index[1] = j;
	  }
	}
    }
  }

  if(type_selected[2] == 1) {
    i_ind = selected_indices[2][0];
    j_ind = selected_indices[2][1];
    k_ind = selected_indices[2][2];
    Q_i = Q->get_Q(i_ind,active_size);
    Q_i_star = Q_star->get_Q(i_ind,active_size);
    Q_k_star_beta = Q_star_beta->get_Q(k_ind,active_size_beta);

    true_k = active_set_beta[k_ind];
    act_set_k = true_act_set[true_k];
    Q_k_star = Q_star->get_Q(act_set_k, active_size);

    nominator_base = y[i_ind]*G[i_ind]-2+(g[i_ind]-2*g_beta[k_ind])/tau;
      nominator_base1 = y[i_ind]*G[i_ind]-2;
    nominator_base2 = g[i_ind]-2*g_beta[k_ind];
    denominator_base = 2*(Q_i[i_ind]+(Q_i_star[i_ind]-4*Q_k_star[i_ind]+4*Q_k_star_beta[k_ind])/tau);
    first_order_criterion = nominator_base+y[j_ind]*G[j_ind]+g[j_ind]/tau;
    for(j = 0; j < active_size; j++) 
      if(y[j] == -1) {
         nominator1 = nominator_base1+y[j]*G[j];
        nominator2 = nominator_base2+g[j];
        nominator = nominator_base+y[j]*G[j]+g[j]/tau;
        if((first_order_criterion < 0 && nominator<-eps && ((nominator1 < -eps) || (nominator2 < -eps))) ||
           (first_order_criterion > 0 && alpha[j] > 1e-8 && nominator > eps && ((nominator1 > eps) || (nominator2 > eps)))) {
	  z = nominator*nominator/(denominator_base + 2*(QD[j]-2*Q_i[j]+(QD_star[j]+2*Q_i_star[j]-4*Q_k_star[j])/tau));
	  if(z > max_z[2]) {
	    max_z[2] = z;
	    best_z_index[2] = j;
	  }
        }
      }
  }

  // choose the best type
  absolute_best_z = -1;
  for(i=0; i<3; i++)
    if((type_selected[i] == 1) && (max_z[i] > absolute_best_z)) {  
      absolute_best_z = max_z[i];
      set_type = i;
      i_out = selected_indices[i][0];
      j_out = best_z_index[i];
      if(i == 2)
	k_out = selected_indices[i][2];
    }

  return 0;
}


bool Solver::be_shrunk(int i, double Gmax1, double Gmax2)
{
	if(is_upper_bound(i))
	{
		if(y[i]==+1)
			return(-G[i] > Gmax1);
		else
			return(-G[i] > Gmax2);
	}
	else if(is_lower_bound(i))
	{
		if(y[i]==+1)
			return(G[i] > Gmax2);
		else
			return(G[i] > Gmax1);
	}
	else
		return(false);
}

bool Solver::be_shrunk_alpha(int i, double max_B1, double max_A1, double max_A2, double min_B1B2, double min_A1A3, double min_A2A4)
{
  int y_i = y[i];
  double deriv_i = y_i*G[i]+g[i]/tau;

  if(alpha[i]<=1e-8) {
    if(y_i == 1 && deriv_i <= max_A1+eps)
      return false;
    if(y_i == -1 && deriv_i <= max_A2+eps)
      return false;
    return deriv_i+min_A1A3+eps>2*max_B1+2;
  }
  else {
    if(y_i == 1) 
      return max_A1-deriv_i<eps && deriv_i-min_A1A3<eps && 2*max_B1+2-deriv_i-min_A2A4<eps && deriv_i+max_A2-2*min_B1B2-2<eps; 
    else 
      return max_A2-deriv_i<eps && deriv_i-min_A2A4<eps && 2*max_B1+2-min_A1A3-deriv_i<eps && max_A1+deriv_i-2*min_B1B2-2<eps;
  }

}

bool Solver::be_shrunk_beta(int i, double max_B1, double max_A1, double max_A2, double min_B1B2, double min_A1A3, double min_A2A4)
{
  double g_beta_i = g_beta[i]/tau;

  if(beta[i]<=1e-8) 
    return (g_beta_i +eps> max_B1 && 2*g_beta_i+2 + eps > max_A1 + max_A2);
  else 
    return (g_beta_i-min_B1B2<eps && max_B1-g_beta_i<eps && 
            2*g_beta_i+2-min_A1A3-min_A2A4<eps && max_A1+max_A2-2*g_beta_i-2<eps);

}

void Solver::do_shrinking()
{
	int i;
	double Gmax1 = -INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }
	double Gmax2 = -INF;		// max { y_i * grad(f)_i | i in I_low(\alpha) }

	// find maximal violating pair first
	for(i=0;i<active_size;i++)
	{
		if(y[i]==+1)
		{
			if(!is_upper_bound(i))
			{
				if(-G[i] >= Gmax1)
					Gmax1 = -G[i];
			}
			if(!is_lower_bound(i))
			{
				if(G[i] >= Gmax2)
					Gmax2 = G[i];
			}
		}
		else
		{
			if(!is_upper_bound(i))
			{
				if(-G[i] >= Gmax2)
					Gmax2 = -G[i];
			}
			if(!is_lower_bound(i))
			{
				if(G[i] >= Gmax1)
					Gmax1 = G[i];
			}
		}
	}

	if(unshrink == false && Gmax1 + Gmax2 <= eps*10)
	{
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
		info("*");
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunk(active_size, Gmax1, Gmax2))
				{
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
}
void Solver::do_shrinking_plus() 
{
  int i, y_i;
  double g_i, alpha_i, deriv_alpha_i;
  double max_B1=-1e20, min_B1B2=1e20, max_A2=-1e20, min_A2A4=1e20, max_A1=-1e20, min_A1A3=1e20;

  // compute all maxima and minima related to alphas
  for(i=0; i<active_size; i++) {
    alpha_i = alpha[i];
    g_i = g[i];
    y_i = y[i];
    deriv_alpha_i = y_i*G[i]+g_i/tau;

    // max A2
    if(alpha_i>1e-8 && y_i==-1 && deriv_alpha_i>max_A2) 
      max_A2 = deriv_alpha_i;

    // min A2A4
    if(y_i==-1 && deriv_alpha_i<min_A2A4) 
      min_A2A4 = deriv_alpha_i;
    
    // max A1
    if(alpha_i>1e-8 && y_i==1 && deriv_alpha_i>max_A1) 
      max_A1 = deriv_alpha_i;
    
    // min A1A3max_A2, min_A2A4, max_A1, min_A1A3
    if(y_i==1 && deriv_alpha_i<min_A1A3) 
      min_A1A3 = deriv_alpha_i;
  }

  // compute all maxima and minima related to betas
  for(i=0; i<active_size_beta; i++) {
    g_i = g_beta[i];

    // max B1
    if(beta[i]>1e-8 && g_i>max_B1) 
      max_B1 = g_i;

    // min B1B2
    if(g_i<min_B1B2) 
      min_B1B2 = g_i;
  }

  if(unshrink == false && max_B1-min_B1B2 < eps*10 &&
     max_A2-min_A2A4 < eps*10 && max_A1 - min_A1A3 < eps*10 &&
     2*max_B1+2-min_A1A3-min_A2A4 < eps*10 && max_A1+max_A2-2*min_B1B2-2 < eps*10) {
    unshrink = true;
    reconstruct_gradient_plus();
    active_size = l;
    active_size_beta = l;
  }

  if(active_size_beta > 2) {
    for(i=0;i<active_size_beta;i++) {
      if(active_size_beta <= 2)
	break;
      if (be_shrunk_beta(i, max_B1, max_A1, max_A2, min_B1B2, min_A1A3, min_A2A4)) {
	active_size_beta--;
	while (active_size_beta > i) {
	  if (!be_shrunk_beta(active_size_beta, max_B1, max_A1, max_A2, min_B1B2, min_A1A3, min_A2A4)) {
	    swap_index_beta(i,active_size_beta);
	    break;
	  }
	  active_size_beta--;
	  if(active_size_beta <= 2)
	    break;
	}
      }
    }
  }

  for(i=0;i<active_size;i++) {
    if (be_shrunk_alpha(i, max_B1, max_A1, max_A2, min_B1B2, min_A1A3, min_A2A4)) {
      active_size--;
      while (active_size > i) {
	if (!be_shrunk_alpha(active_size, max_B1, max_A1, max_A2, min_B1B2, min_A1A3, min_A2A4)) {
	  swap_index_alpha(i,active_size);
	  break;
	}
	active_size--;
      }
    }
  }
}


double Solver::calculate_rho()
{
	double r;
	int nr_free = 0;
	double ub = INF, lb = -INF, sum_free = 0;
	for(int i=0;i<active_size;i++)
	{
		double yG = y[i]*G[i];

		if(is_upper_bound(i))
		{
			if(y[i]==-1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else if(is_lower_bound(i))
		{
			if(y[i]==+1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else
		{
			++nr_free;
			sum_free += yG;
		}
	}

	if(nr_free>0)
		r = sum_free/nr_free;
	else
		r = (ub+lb)/2;

	return r;
}

void Solver::calculate_rho_plus(double& rho, double& rho_star)
{
  int i, pos_size=0, neg_size=0;
  double pos_sum = 0, neg_sum = 0;

  for(i=0; i < active_size; i++)
    if(alpha[i]>1e-8) {
      if(y[i]==1) {
        pos_size++;
        pos_sum += 1-G[i]-g[i]/tau; 
      }
      else {
        neg_size++;
        neg_sum += -1-G[i]+g[i]/tau;
      }
    }

  if(pos_size != 0)
    pos_sum /= pos_size;

  if(neg_size != 0)
    neg_sum /= neg_size;

  rho = (pos_sum+neg_sum)/2;
  rho_star = pos_sum - rho; 
   
}

//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
class Solver_NU: public Solver
{
public:
	Solver_NU() {}
	void Solve(int l, const QMatrix& Q, const double *p, const schar *y,
		   double *alpha, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking)
	{
		this->si = si;
		Solver::Solve(l,Q,p,y,alpha,Cp,Cn,eps,si,shrinking);
	}
private:
	SolutionInfo *si;
	int select_working_set(int &i, int &j);
	double calculate_rho();
	bool be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4);
	void do_shrinking();
};

// return 1 if already optimal, return 0 otherwise
int Solver_NU::select_working_set(int &out_i, int &out_j)
{
	// return i,j such that y_i = y_j and
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	double Gmaxp = -INF;
	double Gmaxp2 = -INF;
	int Gmaxp_idx = -1;

	double Gmaxn = -INF;
	double Gmaxn2 = -INF;
	int Gmaxn_idx = -1;

	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for(int t=0;t<active_size;t++)
		if(y[t]==+1)
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmaxp)
				{
					Gmaxp = -G[t];
					Gmaxp_idx = t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmaxn)
				{
					Gmaxn = G[t];
					Gmaxn_idx = t;
				}
		}

	int ip = Gmaxp_idx;
	int in = Gmaxn_idx;
	const Qfloat *Q_ip = NULL;
	const Qfloat *Q_in = NULL;
	if(ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
		Q_ip = Q->get_Q(ip,active_size);
	if(in != -1)
		Q_in = Q->get_Q(in,active_size);

	for(int j=0;j<active_size;j++)
	{
		if(y[j]==+1)
		{
			if (!is_lower_bound(j))
			{
				double grad_diff=Gmaxp+G[j];
				if (G[j] >= Gmaxp2)
					Gmaxp2 = G[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = QD[ip]+QD[j]-2*Q_ip[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				double grad_diff=Gmaxn-G[j];
				if (-G[j] >= Gmaxn2)
					Gmaxn2 = -G[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = QD[in]+QD[j]-2*Q_in[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	if(max(Gmaxp+Gmaxp2,Gmaxn+Gmaxn2) < eps || Gmin_idx == -1)
		return 1;

	if (y[Gmin_idx] == +1)
		out_i = Gmaxp_idx;
	else
		out_i = Gmaxn_idx;
	out_j = Gmin_idx;

	return 0;
}

bool Solver_NU::be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4)
{
	if(is_upper_bound(i))
	{
		if(y[i]==+1)
			return(-G[i] > Gmax1);
		else
			return(-G[i] > Gmax4);
	}
	else if(is_lower_bound(i))
	{
		if(y[i]==+1)
			return(G[i] > Gmax2);
		else
			return(G[i] > Gmax3);
	}
	else
		return(false);
}

void Solver_NU::do_shrinking()
{
	double Gmax1 = -INF;	// max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
	double Gmax2 = -INF;	// max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
	double Gmax3 = -INF;	// max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
	double Gmax4 = -INF;	// max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

	// find maximal violating pair first
	int i;
	for(i=0;i<active_size;i++)
	{
		if(!is_upper_bound(i))
		{
			if(y[i]==+1)
			{
				if(-G[i] > Gmax1) Gmax1 = -G[i];
			}
			else	if(-G[i] > Gmax4) Gmax4 = -G[i];
		}
		if(!is_lower_bound(i))
		{
			if(y[i]==+1)
			{
				if(G[i] > Gmax2) Gmax2 = G[i];
			}
			else	if(G[i] > Gmax3) Gmax3 = G[i];
		}
	}

	if(unshrink == false && max(Gmax1+Gmax2,Gmax3+Gmax4) <= eps*10)
	{
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4))
				{
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
}

double Solver_NU::calculate_rho()
{
	int nr_free1 = 0,nr_free2 = 0;
	double ub1 = INF, ub2 = INF;
	double lb1 = -INF, lb2 = -INF;
	double sum_free1 = 0, sum_free2 = 0;

	for(int i=0;i<active_size;i++)
	{
		if(y[i]==+1)
		{
			if(is_upper_bound(i))
				lb1 = max(lb1,G[i]);
			else if(is_lower_bound(i))
				ub1 = min(ub1,G[i]);
			else
			{
				++nr_free1;
				sum_free1 += G[i];
			}
		}
		else
		{
			if(is_upper_bound(i))
				lb2 = max(lb2,G[i]);
			else if(is_lower_bound(i))
				ub2 = min(ub2,G[i]);
			else
			{
				++nr_free2;
				sum_free2 += G[i];
			}
		}
	}

	double r1,r2;
	if(nr_free1 > 0)
		r1 = sum_free1/nr_free1;
	else
		r1 = (ub1+lb1)/2;

	if(nr_free2 > 0)
		r2 = sum_free2/nr_free2;
	else
		r2 = (ub2+lb2)/2;

	si->r = (r1+r2)/2;
	return (r1-r2)/2;
}

//
// Q matrices for various formulations
//
class SVC_Q: public Kernel
{
public:
	SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
	:Kernel(prob.l, prob.x, param)
	{
		clone(y,y_,prob.l);
		cache = new Cache(prob.l,(size_t)(param.cache_size*(1<<20)));
		QD = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			QD[i] = (this->*kernel_function)(i,i);
	}

	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;
		int start, j;
		if((start = cache->get_data(i,&data,len)) < len)
		{
#ifdef _OPENMP
#pragma omp parallel for private(j) schedule(guided)
#endif
			for(j=start;j<len;j++)
				data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
		}
		return data;
	}

	double *get_QD() const
	{
		return QD;
	}

	void swap_index(int i, int j) const
	{
		cache->swap_index(i,j);
		Kernel::swap_index(i,j);
		swap(y[i],y[j]);
		swap(QD[i],QD[j]);
	}

	~SVC_Q()
	{
		delete[] y;
		delete cache;
		delete[] QD;
	}
private:
	schar *y;
	Cache *cache;
	double *QD;
};

class ONE_CLASS_Q: public Kernel
{
public:
	ONE_CLASS_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param)
	{
		cache = new Cache(prob.l,(size_t)(param.cache_size*(1<<20)));
		QD = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			QD[i] = (this->*kernel_function)(i,i);
	}

	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;
		int start, j;
		if((start = cache->get_data(i,&data,len)) < len)
		{
			for(j=start;j<len;j++)
				data[j] = (Qfloat)(this->*kernel_function)(i,j);
		}
		return data;
	}

	double *get_QD() const
	{
		return QD;
	}

	void swap_index(int i, int j) const
	{
		cache->swap_index(i,j);
		Kernel::swap_index(i,j);
		swap(QD[i],QD[j]);
	}

	~ONE_CLASS_Q()
	{
		delete cache;
		delete[] QD;
	}
private:
	Cache *cache;
	double *QD;
};

class SVR_Q: public Kernel
{
public:
	SVR_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param)
	{
		l = prob.l;
		cache = new Cache(l,(size_t)(param.cache_size*(1<<20)));
		QD = new double[2*l];
		sign = new schar[2*l];
		index = new int[2*l];
		for(int k=0;k<l;k++)
		{
			sign[k] = 1;
			sign[k+l] = -1;
			index[k] = k;
			index[k+l] = k;
			QD[k] = (this->*kernel_function)(k,k);
			QD[k+l] = QD[k];
		}
		buffer[0] = new Qfloat[2*l];
		buffer[1] = new Qfloat[2*l];
		next_buffer = 0;
	}

	void swap_index(int i, int j) const
	{
		swap(sign[i],sign[j]);
		swap(index[i],index[j]);
		swap(QD[i],QD[j]);
	}

	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;
		int j, real_i = index[i];
		if(cache->get_data(real_i,&data,l) < l)
		{
#ifdef _OPENMP
#pragma omp parallel for private(j) schedule(guided)
#endif
			for(j=0;j<l;j++)
				data[j] = (Qfloat)(this->*kernel_function)(real_i,j);
		}

		// reorder and copy
		Qfloat *buf = buffer[next_buffer];
		next_buffer = 1 - next_buffer;
		schar si = sign[i];
		for(j=0;j<len;j++)
			buf[j] = (Qfloat) si * (Qfloat) sign[j] * data[index[j]];
		return buf;
	}

	double *get_QD() const
	{
		return QD;
	}

	~SVR_Q()
	{
		delete cache;
		delete[] sign;
		delete[] index;
		delete[] buffer[0];
		delete[] buffer[1];
		delete[] QD;
	}
private:
	int l;
	Cache *cache;
	schar *sign;
	int *index;
	mutable int next_buffer;
	Qfloat *buffer[2];
	double *QD;
};

//
// construct and solve various formulations
//
static void solve_c_svc(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn)
{
	int l = prob->l;
	double *minus_ones = new double[l];
	schar *y = new schar[l];

	int i;

	for(i=0;i<l;i++)
	{
		alpha[i] = 0;
		minus_ones[i] = -1;
		if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;
	}

	Solver s;
	s.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
		alpha, Cp, Cn, param->eps, si, param->shrinking);

	double sum_alpha=0;
	for(i=0;i<l;i++)
		sum_alpha += alpha[i];

	if (Cp==Cn)
		info("nu = %f\n", sum_alpha/(Cp*prob->l));

	for(i=0;i<l;i++)
		alpha[i] *= y[i];

	delete[] minus_ones;
	delete[] y;
}

static void solve_w_svm(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn)
{
	int l = prob->l;
	double *minus_ones = new double[l];
	schar *y = new schar[l];

	int i;
	struct svm_model* model;
	if((model=svm_load_model(param->transfer_file_name))==0)
	{
		fprintf(stderr,"can't open model file %s\n",param->transfer_file_name);
		exit(1);
	}


	for(i=0;i<l;i++)
	{
		alpha[i] = 0;
		minus_ones[i] = -1;
		if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;
	}

	Solver s;
	s.Solve_w(l, SVC_Q(*prob,*param,y), minus_ones, y,
		alpha, Cp, Cn, param->eps, si, param->shrinking, model, *prob);

	double sum_alpha=0;
	for(i=0;i<l;i++)
		sum_alpha += alpha[i];

	if (Cp==Cn)
		info("nu = %f\n", sum_alpha/(Cp*prob->l));

	for(i=0;i<l;i++)
		alpha[i] *= y[i];

	delete[] minus_ones;
	delete[] y;
}

static void solve_svm_plus(const svm_problem *prob, const svm_parameter* param,
			   double *alpha, double *beta, Solver::SolutionInfo* si, double Cp, double Cn)
{
	int l = prob->l;
	schar *y = new schar[l];
	schar *y_true = new schar[l];
	svm_parameter cheat_param, cheat_param2; 
	svm_problem cheat_problem, cheat_problem2;
	int i;
	int l_pos=0, l_neg=0;

	// initialize alphas and betas
	for(i=0;i<l;i++) 
	{
		alpha[i] = 0;
		beta[i] = Cp; 
		y[i] = 1;
		if(prob->y[i] > 0) 
		{ 
			y_true[i] = +1; 
			l_pos++;
		}
		else 
		{
			y_true[i]=-1;
			l_neg++;
		}
	}

	cheat_param = *param;
	cheat_param.kernel_type = 2;
	cheat_param.gamma = param->gamma_star;
	cheat_problem = *prob;
	cheat_problem.x = Malloc(struct svm_node*,prob->l);
	memcpy(cheat_problem.x, prob->x_star, l*sizeof(struct svm_node*));



	cheat_param2 = *param;
	cheat_param2.kernel_type = 2;
	cheat_param2.gamma = param->gamma_star;
	cheat_problem2 = *prob;
	cheat_problem2.x = Malloc(struct svm_node*,prob->l);
	memcpy(cheat_problem2.x, prob->x_star, l*sizeof(struct svm_node*));

	SVC_Q kernel1 = SVC_Q(*prob,*param,y);
	SVC_Q kernel2 = SVC_Q(cheat_problem, cheat_param, y);
	SVC_Q kernel3 = SVC_Q(cheat_problem2, cheat_param2, y);
	Solver s;
	
    s.Solve_plus(l, kernel1, kernel2, kernel3, y_true,
		 alpha, beta, Cp, Cn, param->tau, param->eps, si, param->shrinking);

	// produce the same output as SVM
	for(i=0;i<l;i++) { 
		if(alpha[i]<0)
			alpha[i] = 0;
		alpha[i] *= y_true[i];
	}

	delete[] y;
	delete[] y_true;
}


static void solve_nu_svc(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int i;
	int l = prob->l;
	double nu = param->nu;

	schar *y = new schar[l];

	for(i=0;i<l;i++)
		if(prob->y[i]>0)
			y[i] = +1;
		else
			y[i] = -1;

	double sum_pos = nu*l/2;
	double sum_neg = nu*l/2;

	for(i=0;i<l;i++)
		if(y[i] == +1)
		{
			alpha[i] = min(1.0,sum_pos);
			sum_pos -= alpha[i];
		}
		else
		{
			alpha[i] = min(1.0,sum_neg);
			sum_neg -= alpha[i];
		}

	double *zeros = new double[l];

	for(i=0;i<l;i++)
		zeros[i] = 0;

	Solver_NU s;
	s.Solve(l, SVC_Q(*prob,*param,y), zeros, y,
		alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
	double r = si->r;

	info("C = %f\n",1/r);

	for(i=0;i<l;i++)
		alpha[i] *= y[i]/r;

	si->rho /= r;
	si->obj /= (r*r);
	si->upper_bound_p = 1/r;
	si->upper_bound_n = 1/r;

	delete[] y;
	delete[] zeros;
}

static void solve_one_class(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int l = prob->l;
	double *zeros = new double[l];
	schar *ones = new schar[l];
	int i;

	int n = (int)(param->nu*prob->l);	// # of alpha's at upper bound

	for(i=0;i<n;i++)
		alpha[i] = 1;
	if(n<prob->l)
		alpha[n] = param->nu * prob->l - n;
	for(i=n+1;i<l;i++)
		alpha[i] = 0;

	for(i=0;i<l;i++)
	{
		zeros[i] = 0;
		ones[i] = 1;
	}

	Solver s;
	s.Solve(l, ONE_CLASS_Q(*prob,*param), zeros, ones,
		alpha, 1.0, 1.0, param->eps, si, param->shrinking);

	delete[] zeros;
	delete[] ones;
}

static void solve_epsilon_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int l = prob->l;
	double *alpha2 = new double[2*l];
	double *linear_term = new double[2*l];
	schar *y = new schar[2*l];
	int i;

	for(i=0;i<l;i++)
	{
		alpha2[i] = 0;
		linear_term[i] = param->p - prob->y[i];
		y[i] = 1;

		alpha2[i+l] = 0;
		linear_term[i+l] = param->p + prob->y[i];
		y[i+l] = -1;
	}

	Solver s;
	s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
		alpha2, param->C, param->C, param->eps, si, param->shrinking);

	double sum_alpha = 0;
	for(i=0;i<l;i++)
	{
		alpha[i] = alpha2[i] - alpha2[i+l];
		sum_alpha += fabs(alpha[i]);
	}
	info("nu = %f\n",sum_alpha/(param->C*l));

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

static void solve_nu_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int l = prob->l;
	double C = param->C;
	double *alpha2 = new double[2*l];
	double *linear_term = new double[2*l];
	schar *y = new schar[2*l];
	int i;

	double sum = C * param->nu * l / 2;
	for(i=0;i<l;i++)
	{
		alpha2[i] = alpha2[i+l] = min(sum,C);
		sum -= alpha2[i];

		linear_term[i] = - prob->y[i];
		y[i] = 1;

		linear_term[i+l] = prob->y[i];
		y[i+l] = -1;
	}

	Solver_NU s;
	s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
		alpha2, C, C, param->eps, si, param->shrinking);

	info("epsilon = %f\n",-si->r);

	for(i=0;i<l;i++)
		alpha[i] = alpha2[i] - alpha2[i+l];

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

//
// decision_function
//
struct decision_function
{
	double *alpha;
	double *beta;
	double rho;
	double rho_star;
};

static decision_function svm_train_one(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn)
{
	double *alpha = Malloc(double,prob->l);
	double *beta = NULL;
	if(param->svm_type == SVM_PLUS)
		beta = Malloc(double,prob->l);

	Solver::SolutionInfo si;
	switch(param->svm_type)
	{
		case C_SVC:
			solve_c_svc(prob,param,alpha,&si,Cp,Cn);
			break;
		case NU_SVC:
			solve_nu_svc(prob,param,alpha,&si);
			break;
		case ONE_CLASS:
			solve_one_class(prob,param,alpha,&si);
			break;
		case EPSILON_SVR:
			solve_epsilon_svr(prob,param,alpha,&si);
			break;
		case NU_SVR:
			solve_nu_svr(prob,param,alpha,&si);
			break;
		case W_SVM:
			solve_w_svm(prob,param,alpha,&si,Cp,Cn);
			break;
		case SVM_PLUS:
			solve_svm_plus(prob,param,alpha,beta,&si,Cp,Cn);
			break;
	}

	info("obj = %f, rho = %f\n",si.obj,si.rho);

	// output SVs

	int nSV = 0;
	int nBSV = 0;
	int nBSV_star = 0;
	int nSV_star = 0;
	
	for(int i=0;i<prob->l;i++)
	{
		if(fabs(alpha[i]) > 0)
		{
			++nSV;
			if (param->svm_type != SVM_PLUS) 
			{	
				if(prob->y[i] > 0)
				{
					if(fabs(alpha[i]) >= si.upper_bound_p)
						++nBSV;
				}
				else
				{
					if(fabs(alpha[i]) >= si.upper_bound_n)
						++nBSV;
				}
			}
		}
		if (param->svm_type == SVM_PLUS)
		{
			if(fabs(beta[i]) > 0)
				++nSV_star;
		}
	}

	info("nSV = %d, nBSV = %d\n",nSV,nBSV);
	if (param->svm_type == SVM_PLUS)
		info("nSV_star = %d nBSV_star = %d \n",nSV_star,nBSV_star);
	
	decision_function f;
	f.alpha = alpha;
	f.rho = si.rho;
	
	if (param->svm_type == SVM_PLUS) 
	{
		f.beta = beta;
		f.rho_star = si.rho_star;
	}

	return f;
}

// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
static void sigmoid_train(
	int l, const double *dec_values, const double *labels,
	double& A, double& B)
{
	double prior1=0, prior0 = 0;
	int i;

	for (i=0;i<l;i++)
		if (labels[i] > 0) prior1+=1;
		else prior0+=1;

	int max_iter=100;	// Maximal number of iterations
	double min_step=1e-10;	// Minimal step taken in line search
	double sigma=1e-12;	// For numerically strict PD of Hessian
	double eps=1e-5;
	double hiTarget=(prior1+1.0)/(prior1+2.0);
	double loTarget=1/(prior0+2.0);
	double *t=Malloc(double,l);
	double fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
	double newA,newB,newf,d1,d2;
	int iter;

	// Initial Point and Initial Fun Value
	A=0.0; B=log((prior0+1.0)/(prior1+1.0));
	double fval = 0.0;

	for (i=0;i<l;i++)
	{
		if (labels[i]>0) t[i]=hiTarget;
		else t[i]=loTarget;
		fApB = dec_values[i]*A+B;
		if (fApB>=0)
			fval += t[i]*fApB + log(1+exp(-fApB));
		else
			fval += (t[i] - 1)*fApB +log(1+exp(fApB));
	}
	for (iter=0;iter<max_iter;iter++)
	{
		// Update Gradient and Hessian (use H' = H + sigma I)
		h11=sigma; // numerically ensures strict PD
		h22=sigma;
		h21=0.0;g1=0.0;g2=0.0;
		for (i=0;i<l;i++)
		{
			fApB = dec_values[i]*A+B;
			if (fApB >= 0)
			{
				p=exp(-fApB)/(1.0+exp(-fApB));
				q=1.0/(1.0+exp(-fApB));
			}
			else
			{
				p=1.0/(1.0+exp(fApB));
				q=exp(fApB)/(1.0+exp(fApB));
			}
			d2=p*q;
			h11+=dec_values[i]*dec_values[i]*d2;
			h22+=d2;
			h21+=dec_values[i]*d2;
			d1=t[i]-p;
			g1+=dec_values[i]*d1;
			g2+=d1;
		}

		// Stopping Criteria
		if (fabs(g1)<eps && fabs(g2)<eps)
			break;

		// Finding Newton direction: -inv(H') * g
		det=h11*h22-h21*h21;
		dA=-(h22*g1 - h21 * g2) / det;
		dB=-(-h21*g1+ h11 * g2) / det;
		gd=g1*dA+g2*dB;


		stepsize = 1;		// Line Search
		while (stepsize >= min_step)
		{
			newA = A + stepsize * dA;
			newB = B + stepsize * dB;

			// New function value
			newf = 0.0;
			for (i=0;i<l;i++)
			{
				fApB = dec_values[i]*newA+newB;
				if (fApB >= 0)
					newf += t[i]*fApB + log(1+exp(-fApB));
				else
					newf += (t[i] - 1)*fApB +log(1+exp(fApB));
			}
			// Check sufficient decrease
			if (newf<fval+0.0001*stepsize*gd)
			{
				A=newA;B=newB;fval=newf;
				break;
			}
			else
				stepsize = stepsize / 2.0;
		}

		if (stepsize < min_step)
		{
			info("Line search fails in two-class probability estimates\n");
			break;
		}
	}

	if (iter>=max_iter)
		info("Reaching maximal iterations in two-class probability estimates\n");
	free(t);
}

static double sigmoid_predict(double decision_value, double A, double B)
{
	double fApB = decision_value*A+B;
	// 1-p used later; avoid catastrophic cancellation
	if (fApB >= 0)
		return exp(-fApB)/(1.0+exp(-fApB));
	else
		return 1.0/(1+exp(fApB)) ;
}

// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng to predict probabilities
static void multiclass_probability(int k, double **r, double *p)
{
	int t,j;
	int iter = 0, max_iter=max(100,k);
	double **Q=Malloc(double *,k);
	double *Qp=Malloc(double,k);
	double pQp, eps=0.005/k;

	for (t=0;t<k;t++)
	{
		p[t]=1.0/k;  // Valid if k = 1
		Q[t]=Malloc(double,k);
		Q[t][t]=0;
		for (j=0;j<t;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=Q[j][t];
		}
		for (j=t+1;j<k;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=-r[j][t]*r[t][j];
		}
	}
	for (iter=0;iter<max_iter;iter++)
	{
		// stopping condition, recalculate QP,pQP for numerical accuracy
		pQp=0;
		for (t=0;t<k;t++)
		{
			Qp[t]=0;
			for (j=0;j<k;j++)
				Qp[t]+=Q[t][j]*p[j];
			pQp+=p[t]*Qp[t];
		}
		double max_error=0;
		for (t=0;t<k;t++)
		{
			double error=fabs(Qp[t]-pQp);
			if (error>max_error)
				max_error=error;
		}
		if (max_error<eps) break;

		for (t=0;t<k;t++)
		{
			double diff=(-Qp[t]+pQp)/Q[t][t];
			p[t]+=diff;
			pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/(1+diff)/(1+diff);
			for (j=0;j<k;j++)
			{
				Qp[j]=(Qp[j]+diff*Q[t][j])/(1+diff);
				p[j]/=(1+diff);
			}
		}
	}
	if (iter>=max_iter)
		info("Exceeds max_iter in multiclass_prob\n");
	for(t=0;t<k;t++) free(Q[t]);
	free(Q);
	free(Qp);
}

// Using cross-validation decision values to get parameters for SVC probability estimates
static void svm_binary_svc_probability(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn, double& probA, double& probB)
{
	int i;
	int nr_fold = 5;
	int *perm = Malloc(int,prob->l);
	double *dec_values = Malloc(double,prob->l);

	// random shuffle
	for(i=0;i<prob->l;i++) perm[i]=i;
	for(i=0;i<prob->l;i++)
	{
		int j = i+rand()%(prob->l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<nr_fold;i++)
	{
		int begin = i*prob->l/nr_fold;
		int end = (i+1)*prob->l/nr_fold;
		int j,k;
		struct svm_problem subprob;

		subprob.l = prob->l-(end-begin);
		subprob.x = Malloc(struct svm_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<prob->l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		int p_count=0,n_count=0;
		for(j=0;j<k;j++)
			if(subprob.y[j]>0)
				p_count++;
			else
				n_count++;

		if(p_count==0 && n_count==0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = 0;
		else if(p_count > 0 && n_count == 0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = 1;
		else if(p_count == 0 && n_count > 0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = -1;
		else
		{
			svm_parameter subparam = *param;
			subparam.probability=0;
			subparam.C=1.0;
			subparam.nr_weight=2;
			subparam.weight_label = Malloc(int,2);
			subparam.weight = Malloc(double,2);
			subparam.weight_label[0]=+1;
			subparam.weight_label[1]=-1;
			subparam.weight[0]=Cp;
			subparam.weight[1]=Cn;
			struct svm_model *submodel = svm_train(&subprob,&subparam);
			for(j=begin;j<end;j++)
			{
				svm_predict_values(submodel,prob->x[perm[j]],&(dec_values[perm[j]]));
				// ensure +1 -1 order; reason not using CV subroutine
				dec_values[perm[j]] *= submodel->label[0];
			}
			svm_free_and_destroy_model(&submodel);
			svm_destroy_param(&subparam);
		}
		free(subprob.x);
		free(subprob.y);
	}
	sigmoid_train(prob->l,dec_values,prob->y,probA,probB);
	free(dec_values);
	free(perm);
}

// Binning method from the oneclass_prob paper by Que and Lin to predict the probability as a normal instance (i.e., not an outlier)
static double predict_one_class_probability(const svm_model *model, double dec_value)
{
	double prob_estimate = 0.0;
	int nr_marks = 10;

	if(dec_value < model->prob_density_marks[0])
		prob_estimate = 0.001;
	else if(dec_value > model->prob_density_marks[nr_marks-1])
		prob_estimate = 0.999;
	else
	{
		for(int i=1;i<nr_marks;i++)
			if(dec_value < model->prob_density_marks[i])
			{
				prob_estimate = (double)i/nr_marks;
				break;
			}
	}
	return prob_estimate;
}

static int compare_double(const void *a, const void *b)
{
	if(*(double *)a > *(double *)b)
		return 1;
	else if(*(double *)a < *(double *)b)
		return -1;
	return 0;
}

// Get parameters for one-class SVM probability estimates
static int svm_one_class_probability(const svm_problem *prob, const svm_model *model, double *prob_density_marks)
{
	double *dec_values = Malloc(double,prob->l);
	double *pred_results = Malloc(double,prob->l);
	int ret = 0;
	int nr_marks = 10;

	for(int i=0;i<prob->l;i++)
		pred_results[i] = svm_predict_values(model,prob->x[i],&dec_values[i]);
	qsort(dec_values,prob->l,sizeof(double),compare_double);

	int neg_counter=0;
	for(int i=0;i<prob->l;i++)
		if(dec_values[i]>=0)
		{
			neg_counter = i;
			break;
		}

	int pos_counter = prob->l-neg_counter;
	if(neg_counter<nr_marks/2 || pos_counter<nr_marks/2)
	{
		fprintf(stderr,"WARNING: number of positive or negative decision values <%d; too few to do a probability estimation.\n",nr_marks/2);
		ret = -1;
	}
	else
	{
		// Binning by density
		double *tmp_marks = Malloc(double,nr_marks+1);
		int mid = nr_marks/2;
		for(int i=0;i<mid;i++)
			tmp_marks[i] = dec_values[i*neg_counter/mid];
		tmp_marks[mid] = 0;
		for(int i=mid+1;i<nr_marks+1;i++)
			tmp_marks[i] = dec_values[neg_counter-1+(i-mid)*pos_counter/mid];

		for(int i=0;i<nr_marks;i++)
			prob_density_marks[i] = (tmp_marks[i]+tmp_marks[i+1])/2;
		free(tmp_marks);
	}
	free(dec_values);
	free(pred_results);
	return ret;
}

// Return parameter of a Laplace distribution
static double svm_svr_probability(
	const svm_problem *prob, const svm_parameter *param)
{
	int i;
	int nr_fold = 5;
	double *ymv = Malloc(double,prob->l);
	double mae = 0;

	svm_parameter newparam = *param;
	newparam.probability = 0;
	svm_cross_validation(prob,&newparam,nr_fold,ymv);
	for(i=0;i<prob->l;i++)
	{
		ymv[i]=prob->y[i]-ymv[i];
		mae += fabs(ymv[i]);
	}
	mae /= prob->l;
	double std=sqrt(2*mae*mae);
	int count=0;
	mae=0;
	for(i=0;i<prob->l;i++)
		if (fabs(ymv[i]) > 5*std)
			count=count+1;
		else
			mae+=fabs(ymv[i]);
	mae /= (prob->l-count);
	info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= %g\n",mae);
	free(ymv);
	return mae;
}


// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	//
	// Labels are ordered by their first occurrence in the training set.
	// However, for two-class sets with -1/+1 labels and -1 appears first,
	// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
	//
	if (nr_class == 2 && label[0] == -1 && label[1] == 1)
	{
		swap(label[0],label[1]);
		swap(count[0],count[1]);
		for(i=0;i<l;i++)
		{
			if(data_label[i] == 0)
				data_label[i] = 1;
			else
				data_label[i] = 0;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

//
// Interface functions
//
svm_model *svm_train(const svm_problem *prob, const svm_parameter *param)
{
	
	svm_model *model = Malloc(svm_model,1);
	model->param = *param;
	model->free_sv = 0;	// XXX

	if(param->svm_type == ONE_CLASS ||
	   param->svm_type == EPSILON_SVR ||
	   param->svm_type == NU_SVR)
	{
		// regression or one-class-svm
		model->nr_class = 2;
		model->label = NULL;
		model->nSV = NULL;
		model->probA = NULL; model->probB = NULL;
		model->prob_density_marks = NULL;
		model->sv_coef = Malloc(double *,1);

		
		decision_function f = svm_train_one(prob,param,0,0);
		model->rho = Malloc(double,1);
		model->rho[0] = f.rho;

		int nSV = 0;
		int i;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0) ++nSV;
		model->l = nSV;
		model->SV = Malloc(svm_node *,nSV);
		model->sv_coef[0] = Malloc(double,nSV);
		model->sv_indices = Malloc(int,nSV);
		int j = 0;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0)
			{
				model->SV[j] = prob->x[i];
				model->sv_coef[0][j] = f.alpha[i];
				model->sv_indices[j] = i+1;
				++j;
			}

		if(param->probability &&
		   (param->svm_type == EPSILON_SVR ||
		    param->svm_type == NU_SVR))
		{
			model->probA = Malloc(double,1);
			model->probA[0] = svm_svr_probability(prob,param);
		}
		else if(param->probability && param->svm_type == ONE_CLASS)
		{
			int nr_marks = 10;
			double *prob_density_marks = Malloc(double,nr_marks);

			if(svm_one_class_probability(prob,model,prob_density_marks) == 0)
				model->prob_density_marks = prob_density_marks;
			else
				free(prob_density_marks);
		}

		free(f.alpha);
	}
	else
	{
		// classification
		int l = prob->l;
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);
		if(nr_class == 1)
			info("WARNING: training data in only one class. See README for details.\n");

		svm_node **x = Malloc(svm_node *,l);
		svm_node **x_star = NULL;
		int i;
		for(i=0;i<l;i++)
		{
			x[i] = prob->x[perm[i]];
			// print x[i]->value
			// std::cout << x[i]->value << std::endl;
		}

		if(param->svm_type == SVM_PLUS)
		{
			x_star = Malloc(svm_node *,l);
			for(i=0;i<l;i++)
				x_star[i] = prob->x_star[perm[i]];
		}

		// calculate weighted C
		double *weighted_C = Malloc(double, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;
		for(i=0;i<param->nr_weight;i++)
		{
			int j;
			for(j=0;j<nr_class;j++)
				if(param->weight_label[i] == label[j])
					break;
			if(j == nr_class)
				fprintf(stderr,"WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
			else
				weighted_C[j] *= param->weight[i];
		}

		// train k*(k-1)/2 models
		
		bool *nonzero = Malloc(bool,l);
		bool *nonzero_star = Malloc(bool,l);
		for(i=0;i<l;i++) 
		{
			nonzero[i] = false;
			nonzero_star[i] = false;
		}
			
		decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);
		
		double *probA=NULL,*probB=NULL;
		if (param->probability)
		{
			probA=Malloc(double,nr_class*(nr_class-1)/2);
			probB=Malloc(double,nr_class*(nr_class-1)/2);
		}

		int p = 0, p_star = 0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				svm_problem sub_prob;
				int si = start[i], sj = start[j];
				int ci = count[i], cj = count[j];
				sub_prob.l = ci+cj;
				sub_prob.x = Malloc(svm_node *,sub_prob.l);
				sub_prob.y = Malloc(double,sub_prob.l);
				if(param->svm_type == SVM_PLUS)
					sub_prob.x_star = Malloc(svm_node *,sub_prob.l);    				
				int k;
				for(k=0;k<ci;k++)
				{
					sub_prob.x[k] = x[si+k];
					sub_prob.y[k] = +1;
					if(param->svm_type == SVM_PLUS)
		  				sub_prob.x_star[k] = x_star[si+k];
				}
				for(k=0;k<cj;k++)
				{
					sub_prob.x[ci+k] = x[sj+k];
					sub_prob.y[ci+k] = -1;
					if(param->svm_type == SVM_PLUS)
		  				sub_prob.x_star[ci+k] = x_star[sj+k];					
				}

				if(param->probability)
					svm_binary_svc_probability(&sub_prob,param,weighted_C[i],weighted_C[j],probA[p],probB[p]);
				
				f[p] = svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);

				for(k=0;k<ci;k++) 
				{
					if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
						nonzero[si+k] = true;
					if(param->svm_type == SVM_PLUS) 
					{
						if(!nonzero_star[si+k] && f[p].beta[k] > 0) 
							nonzero_star[si+k] = true;
					}
				}
				for(k=0;k<cj;k++)
				{
					if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
						nonzero[sj+k] = true;
					if(param->svm_type == SVM_PLUS)
					{
						if(!nonzero_star[sj+k] && f[p].beta[ci+k] > 0) 
		  					nonzero_star[sj+k] = true;
					}
				}
				free(sub_prob.x);
				if(param->svm_type == SVM_PLUS)
	      			free(sub_prob.x_star);
				free(sub_prob.y);
				++p;
			}

		// build output

		model->nr_class = nr_class;

		model->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model->label[i] = label[i];

		model->rho = Malloc(double,nr_class*(nr_class-1)/2);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			model->rho[i] = f[i].rho;

    	if(param->svm_type == SVM_PLUS) 
		{
			model->rho_star = Malloc(double,nr_class*(nr_class-1)/2);
			for(i=0;i<nr_class*(nr_class-1)/2;i++)
	  			model->rho_star[i] = f[i].rho_star;
    	}		

		if(param->probability)
		{
			model->probA = Malloc(double,nr_class*(nr_class-1)/2);
			model->probB = Malloc(double,nr_class*(nr_class-1)/2);
			for(i=0;i<nr_class*(nr_class-1)/2;i++)
			{
				model->probA[i] = probA[i];
				model->probB[i] = probB[i];
			}
		}
		else
		{
			model->probA=NULL;
			model->probB=NULL;
		}
		model->prob_density_marks=NULL;	// for one-class SVM probabilistic outputs only

		int total_sv = 0;
		int total_sv_star = 0;
		int *nz_count = Malloc(int,nr_class);
		int *nz_count_star = Malloc(int,nr_class);

		model->nSV = Malloc(int,nr_class);
		model->nSV_star = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
		{
			int nSV = 0;
			int nSV_star = 0;
			for(int j=0;j<count[i];j++)
			{
				if(nonzero[start[i]+j])
				{
					++nSV;
					++total_sv;
				}
				if(nonzero_star[start[i]+j])
				{
					++nSV_star;
					++total_sv_star;
				}
			}
			model->nSV[i] = nSV;
			nz_count[i] = nSV;
			model->nSV_star[i] = nSV_star;
	  		nz_count_star[i] = nSV_star;
		}

		info("Total nSV = %d\n",total_sv);
		info("Total nSV_star = %d\n",total_sv_star);

		model->l = total_sv;
		model->SV = Malloc(svm_node *,total_sv);
		model->sv_indices = Malloc(int,total_sv);
		p = 0;
		for(i=0;i<l;i++)
			if(nonzero[i])
			{
				model->SV[p] = x[i];
				model->sv_indices[p++] = perm[i] + 1;
			}
		
		if(model->param.svm_type == SVM_PLUS)
			{
				model->SV_star = Malloc(svm_node *,total_sv_star);
				model->sv_indices_star = Malloc(int,total_sv_star);
				p_star = 0;
				for(i=0;i<l;i++)
					if(nonzero_star[i]) 
					{
						model->SV_star[p_star] = x_star[i];
						model->sv_indices_star[p_star++] = perm[i] + 1;
					}
			}

		int *nz_start = Malloc(int,nr_class);
		nz_start[0] = 0;
		for(i=1;i<nr_class;i++)
			nz_start[i] = nz_start[i-1]+nz_count[i-1];

		model->sv_coef = Malloc(double *,nr_class-1);
		for(i=0;i<nr_class-1;i++)
			model->sv_coef[i] = Malloc(double,total_sv);

		int *nz_start_star=NULL;

		if(model->param.svm_type == SVM_PLUS)
		{
			nz_start_star = Malloc(int,nr_class);
			nz_start_star[0] = 0;
			for(i=1;i<nr_class;i++)
				nz_start_star[i] = nz_start_star[i-1]+nz_count_star[i-1];

			model->sv_coef_star = Malloc(double *,nr_class-1);
			for(i=0;i<nr_class-1;i++)
				model->sv_coef_star[i] = Malloc(double,total_sv_star);
		}
		p = 0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				// classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]

				int si = start[i];
				int sj = start[j];
				int ci = count[i];
				int cj = count[j];

				int q = nz_start[i];
				int k;
				int q_star = 0;
				if(model->param.svm_type == SVM_PLUS)
					q_star = nz_start_star[i];

				for(k=0;k<ci;k++)
				{
					if(nonzero[si+k])
						model->sv_coef[j-1][q++] = f[p].alpha[k];
					if(model->param.svm_type == SVM_PLUS)
						if(nonzero_star[si+k])
							model->sv_coef_star[j-1][q_star++] = f[p].beta[k];
				}

				q = nz_start[j];
				if(model->param.svm_type == SVM_PLUS)
					q_star = nz_start_star[j];
				for(k=0;k<cj;k++)
					if(nonzero[sj+k])
						model->sv_coef[i][q++] = f[p].alpha[ci+k];
					if(model->param.svm_type == SVM_PLUS)
						if(nonzero_star[sj+k])
							model->sv_coef_star[i][q_star++] = f[p].beta[ci+k];
				++p;
			}

		free(label);
		free(probA);
		free(probB);
		free(count);
		free(perm);
		free(start);
		free(x);
		free(weighted_C);
		free(nonzero);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			free(f[i].alpha);
		free(nz_count);
		free(nz_start);
		if (model->param.svm_type == SVM_PLUS)
		{
			free(nz_count_star);
			free(nz_start_star);
		}
		free(f);
	}
	return model;
}

// Stratified cross validation
void svm_cross_validation(const svm_problem *prob, const svm_parameter *param, int nr_fold, double *target)
{
	int i;
	int *fold_start;
	int l = prob->l;
	int *perm = Malloc(int,l);
	int nr_class;
	if (nr_fold > l)
	{
		fprintf(stderr,"WARNING: # folds (%d) > # data (%d). Will use # folds = # data instead (i.e., leave-one-out cross validation)\n", nr_fold, l);
		nr_fold = l;
	}
	fold_start = Malloc(int,nr_fold+1);
	// stratified cv may not give leave-one-out rate
	// Each class to l folds -> some folds may have zero elements
	if((param->svm_type == C_SVC ||
	    param->svm_type == NU_SVC) && nr_fold < l)
	{
		int *start = NULL;
		int *label = NULL;
		int *count = NULL;
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);

		// random shuffle and then data grouped by fold using the array perm
		int *fold_count = Malloc(int,nr_fold);
		int c;
		int *index = Malloc(int,l);
		for(i=0;i<l;i++)
			index[i]=perm[i];
		for (c=0; c<nr_class; c++)
			for(i=0;i<count[c];i++)
			{
				int j = i+rand()%(count[c]-i);
				swap(index[start[c]+j],index[start[c]+i]);
			}
		for(i=0;i<nr_fold;i++)
		{
			fold_count[i] = 0;
			for (c=0; c<nr_class;c++)
				fold_count[i]+=(i+1)*count[c]/nr_fold-i*count[c]/nr_fold;
		}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		for (c=0; c<nr_class;c++)
			for(i=0;i<nr_fold;i++)
			{
				int begin = start[c]+i*count[c]/nr_fold;
				int end = start[c]+(i+1)*count[c]/nr_fold;
				for(int j=begin;j<end;j++)
				{
					perm[fold_start[i]] = index[j];
					fold_start[i]++;
				}
			}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		free(start);
		free(label);
		free(count);
		free(index);
		free(fold_count);
	}
	else
	{
		for(i=0;i<l;i++) perm[i]=i;
		for(i=0;i<l;i++)
		{
			int j = i+rand()%(l-i);
			swap(perm[i],perm[j]);
		}
		for(i=0;i<=nr_fold;i++)
			fold_start[i]=i*l/nr_fold;
	}

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct svm_problem subprob;

		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct svm_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct svm_model *submodel = svm_train(&subprob,param);
		if(param->probability &&
		   (param->svm_type == C_SVC || param->svm_type == NU_SVC))
		{
			double *prob_estimates=Malloc(double,svm_get_nr_class(submodel));
			for(j=begin;j<end;j++)
				target[perm[j]] = svm_predict_probability(submodel,prob->x[perm[j]],prob_estimates);
			free(prob_estimates);
		}
		else
			for(j=begin;j<end;j++)
				target[perm[j]] = svm_predict(submodel,prob->x[perm[j]]);
		svm_free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
	}
	free(fold_start);
	free(perm);
}


int svm_get_svm_type(const svm_model *model)
{
	return model->param.svm_type;
}

int svm_get_nr_class(const svm_model *model)
{
	return model->nr_class;
}

void svm_get_labels(const svm_model *model, int* label)
{
	if (model->label != NULL)
		for(int i=0;i<model->nr_class;i++)
			label[i] = model->label[i];
}

void svm_get_sv_indices(const svm_model *model, int* indices)
{
	if (model->sv_indices != NULL)
		for(int i=0;i<model->l;i++)
			indices[i] = model->sv_indices[i];
}

int svm_get_nr_sv(const svm_model *model)
{
	return model->l;
}

double svm_get_svr_probability(const svm_model *model)
{
	if ((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
	    model->probA!=NULL)
		return model->probA[0];
	else
	{
		fprintf(stderr,"Model doesn't contain information for SVR probability inference\n");
		return 0;
	}
}

double svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values)
{
	int i;
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
	{
		double *sv_coef = model->sv_coef[0];
		double sum = 0;
#ifdef _OPENMP
#pragma omp parallel for private(i) reduction(+:sum) schedule(guided)
#endif
		for(i=0;i<model->l;i++)
			sum += sv_coef[i] * Kernel::k_function(x,model->SV[i],model->param);
		sum -= model->rho[0];
		*dec_values = sum;

		if(model->param.svm_type == ONE_CLASS)
			return (sum>0)?1:-1;
		else
			return sum;
	}
	else if(model->param.svm_type == W_SVM)
	{
		
		int nr_class = model->nr_class;
		int l = model->l;
		
		double t_sum = 0;
		

		double *kvalue = Malloc(double,l);

		struct svm_model* t_model;
		if((t_model=svm_load_model(model->param.transfer_file_name))==0)
		{
			fprintf(stderr,"can't open model file %s\n",model->param.transfer_file_name);
			exit(1);
		}
		int t = t_model->l;

		double *kvalue2 = Malloc(double,t);

#ifdef _OPENMP
#pragma omp parallel for private(i) schedule(guided)
#endif
		for(i=0;i<l;i++)
			kvalue[i] = Kernel::k_function(x,model->SV[i],model->param);
		
		
		for (i=0;i<t;i++)
			kvalue2[i]= Kernel::k_function(x,t_model->SV[i],t_model->param);

		int *start = Malloc(int,nr_class);
		start[0] = 0;
		for(i=1;i<nr_class;i++)
			start[i] = start[i-1]+model->nSV[i-1];

		int *start2 = Malloc(int,model->nr_class);
		start2[0] = 0;
		for(i=1;i<model->nr_class;i++)
			start2[i] = start2[i-1]+t_model->nSV[i-1];

		int *vote = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			vote[i] = 0;

		int p=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				double sum = 0;
				int si = start[i];
				int sj = start[j];
				int ci = model->nSV[i];
				int cj = model->nSV[j];

				int k;
				double *coef1 = model->sv_coef[j-1];
				double *coef2 = model->sv_coef[i];
				for(k=0;k<ci;k++)
					sum += coef1[si+k] * kvalue[si+k];
				for(k=0;k<cj;k++)
					sum += coef2[sj+k] * kvalue[sj+k];
				sum -= model->rho[p];

				int si2 = start2[i];
				int sj2 = start2[j];
				int ci2 = t_model->nSV[i];
				int cj2 = t_model->nSV[j];

				double *coef12 = t_model->sv_coef[j-1];
				double *coef22 = t_model->sv_coef[i];
				for(k=0;k<ci2;k++)
					sum += coef12[si2+k] * kvalue2[si2+k];
				for(k=0;k<cj2;k++)
					sum += coef22[sj2+k] * kvalue2[sj2+k];
				sum -= t_model->rho[p];
				
				dec_values[p] = sum;

				if(dec_values[p] > 0)
					++vote[i];
				else
					++vote[j];
				p++;
			}

		int vote_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(vote[i] > vote[vote_max_idx])
				vote_max_idx = i;

		free(kvalue);
		free(start);
		free(vote);
		return model->label[vote_max_idx];
	}
	else
	{
		int nr_class = model->nr_class;
		int l = model->l;

		double *kvalue = Malloc(double,l);
#ifdef _OPENMP
#pragma omp parallel for private(i) schedule(guided)
#endif
		for(i=0;i<l;i++)
			kvalue[i] = Kernel::k_function(x,model->SV[i],model->param);

		int *start = Malloc(int,nr_class);
		start[0] = 0;
		for(i=1;i<nr_class;i++)
			start[i] = start[i-1]+model->nSV[i-1];

		int *vote = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			vote[i] = 0;

		int p=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				double sum = 0;
				int si = start[i];
				int sj = start[j];
				int ci = model->nSV[i];
				int cj = model->nSV[j];

				int k;
				double *coef1 = model->sv_coef[j-1];
				double *coef2 = model->sv_coef[i];
				for(k=0;k<ci;k++)
					sum += coef1[si+k] * kvalue[si+k];
				for(k=0;k<cj;k++)
					sum += coef2[sj+k] * kvalue[sj+k];
				sum -= model->rho[p];
				dec_values[p] = sum;

				if(dec_values[p] > 0)
					++vote[i];
				else
					++vote[j];
				p++;
			}

		int vote_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(vote[i] > vote[vote_max_idx])
				vote_max_idx = i;

		free(kvalue);
		free(start);
		free(vote);
		return model->label[vote_max_idx];
	}
}

double svm_predict(const svm_model *model, const svm_node *x)
{
	int nr_class = model->nr_class;
	double *dec_values;
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
		dec_values = Malloc(double, 1);
	else
		dec_values = Malloc(double, nr_class*(nr_class-1)/2);
	double pred_result = svm_predict_values(model, x, dec_values);
	free(dec_values);
	return pred_result;
}

double svm_predict_probability(
	const svm_model *model, const svm_node *x, double *prob_estimates)
{
	if ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
	    model->probA!=NULL && model->probB!=NULL)
	{
		int i;
		int nr_class = model->nr_class;
		double *dec_values = Malloc(double, nr_class*(nr_class-1)/2);
		svm_predict_values(model, x, dec_values);

		double min_prob=1e-7;
		double **pairwise_prob=Malloc(double *,nr_class);
		for(i=0;i<nr_class;i++)
			pairwise_prob[i]=Malloc(double,nr_class);
		int k=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				pairwise_prob[i][j]=min(max(sigmoid_predict(dec_values[k],model->probA[k],model->probB[k]),min_prob),1-min_prob);
				pairwise_prob[j][i]=1-pairwise_prob[i][j];
				k++;
			}
		if (nr_class == 2)
		{
			prob_estimates[0] = pairwise_prob[0][1];
			prob_estimates[1] = pairwise_prob[1][0];
		}
		else
			multiclass_probability(nr_class,pairwise_prob,prob_estimates);

		int prob_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(prob_estimates[i] > prob_estimates[prob_max_idx])
				prob_max_idx = i;
		for(i=0;i<nr_class;i++)
			free(pairwise_prob[i]);
		free(dec_values);
		free(pairwise_prob);
		return model->label[prob_max_idx];
	}
	else if(model->param.svm_type == ONE_CLASS && model->prob_density_marks!=NULL)
	{
		double dec_value;
		double pred_result = svm_predict_values(model,x,&dec_value);
		prob_estimates[0] = predict_one_class_probability(model,dec_value);
		prob_estimates[1] = 1-prob_estimates[0];
		return pred_result;
	}
	else
		return svm_predict(model, x);
}

static const char *svm_type_table[] =
{
	"c_svc","nu_svc","one_class","epsilon_svr","nu_svr","w_svm",NULL
};

static const char *kernel_type_table[]=
{
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};

int svm_save_model(const char *model_file_name, const svm_model *model)
{
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale) {
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	const svm_parameter& param = model->param;

	fprintf(fp,"svm_type %s\n", svm_type_table[param.svm_type]);
	fprintf(fp,"kernel_type %s\n", kernel_type_table[param.kernel_type]);

	if(param.svm_type == W_SVM)
		fprintf(fp,"t_model %s\n", param.transfer_file_name);

	if(param.kernel_type == POLY)
		fprintf(fp,"degree %d\n", param.degree);

	if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
		fprintf(fp,"gamma %.17g\n", param.gamma);

	if(param.kernel_type == POLY || param.kernel_type == SIGMOID)
		fprintf(fp,"coef0 %.17g\n", param.coef0);

	int nr_class = model->nr_class;
	int l = model->l;
	fprintf(fp, "nr_class %d\n", nr_class);
	fprintf(fp, "total_sv %d\n",l);

	{
		fprintf(fp, "rho");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %.17g",model->rho[i]);
		fprintf(fp, "\n");
	}

	if(model->label)
	{
		fprintf(fp, "label");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->label[i]);
		fprintf(fp, "\n");
	}

	if(model->probA) // regression has probA only
	{
		fprintf(fp, "probA");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %.17g",model->probA[i]);
		fprintf(fp, "\n");
	}
	if(model->probB)
	{
		fprintf(fp, "probB");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %.17g",model->probB[i]);
		fprintf(fp, "\n");
	}
	if(model->prob_density_marks)
	{
		fprintf(fp, "prob_density_marks");
		int nr_marks=10;
		for(int i=0;i<nr_marks;i++)
			fprintf(fp," %.17g",model->prob_density_marks[i]);
		fprintf(fp, "\n");
	}

	if(model->nSV)
	{
		fprintf(fp, "nr_sv");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->nSV[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "SV\n");
	const double * const *sv_coef = model->sv_coef;
	const svm_node * const *SV = model->SV;

	for(int i=0;i<l;i++)
	{
		for(int j=0;j<nr_class-1;j++)
			fprintf(fp, "%.17g ",sv_coef[j][i]);

		const svm_node *p = SV[i];

		if(param.kernel_type == PRECOMPUTED)
			fprintf(fp,"0:%d ",(int)(p->value));
		else
			while(p->index != -1)
			{
				fprintf(fp,"%d:%.8g ",p->index,p->value);
				p++;
			}
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var) do{ if (fscanf(_stream, _format, _var) != 1) return false; }while(0)
bool read_model_header(FILE *fp, svm_model* model)
{
	svm_parameter& param = model->param;
	// parameters for training only won't be assigned, but arrays are assigned as NULL for safety
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	char cmd[81];
	while(1)
	{
		FSCANF(fp,"%80s",cmd);

		if(strcmp(cmd,"svm_type")==0)
		{
			FSCANF(fp,"%80s",cmd);
			int i;
			for(i=0;svm_type_table[i];i++)
			{
				if(strcmp(svm_type_table[i],cmd)==0)
				{
					param.svm_type=i;
					break;
				}
			}
			if(svm_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown svm type.\n");
				return false;
			}
		}
		else if(strcmp(cmd,"kernel_type")==0)
		{
			FSCANF(fp,"%80s",cmd);
			int i;
			for(i=0;kernel_type_table[i];i++)
			{
				if(strcmp(kernel_type_table[i],cmd)==0)
				{
					param.kernel_type=i;
					break;
				}
			}
			if(kernel_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown kernel function.\n");
				return false;
			}
		}
		else if(strcmp(cmd,"t_model")==0)
		{
			// transfer model file name
			FSCANF(fp,"%80s",&param.transfer_file_name);
		}
		else if(strcmp(cmd,"degree")==0)
			FSCANF(fp,"%d",&param.degree);
		else if(strcmp(cmd,"gamma")==0)
			FSCANF(fp,"%lf",&param.gamma);
		else if(strcmp(cmd,"coef0")==0)
			FSCANF(fp,"%lf",&param.coef0);
		else if(strcmp(cmd,"nr_class")==0)
			FSCANF(fp,"%d",&model->nr_class);
		else if(strcmp(cmd,"total_sv")==0)
			FSCANF(fp,"%d",&model->l);
		else if(strcmp(cmd,"rho")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->rho = Malloc(double,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%lf",&model->rho[i]);
		}
		else if(strcmp(cmd,"label")==0)
		{
			int n = model->nr_class;
			model->label = Malloc(int,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%d",&model->label[i]);
		}
		else if(strcmp(cmd,"probA")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probA = Malloc(double,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%lf",&model->probA[i]);
		}
		else if(strcmp(cmd,"probB")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probB = Malloc(double,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%lf",&model->probB[i]);
		}
		else if(strcmp(cmd,"prob_density_marks")==0)
		{
			int n = 10;	// nr_marks
			model->prob_density_marks = Malloc(double,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%lf",&model->prob_density_marks[i]);
		}
		else if(strcmp(cmd,"nr_sv")==0)
		{
			int n = model->nr_class;
			model->nSV = Malloc(int,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%d",&model->nSV[i]);
		}
		else if(strcmp(cmd,"SV")==0)
		{
			while(1)
			{
				int c = getc(fp);
				if(c==EOF || c=='\n') break;
			}
			break;
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			return false;
		}
	}

	return true;

}

svm_model *svm_load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"rb");
	if(fp==NULL) return NULL;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale) {
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	// read parameters

	svm_model *model = Malloc(svm_model,1);
	model->rho = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->prob_density_marks = NULL;
	model->sv_indices = NULL;
	model->label = NULL;
	model->nSV = NULL;

	// read header
	if (!read_model_header(fp, model))
	{
		fprintf(stderr, "ERROR: fscanf failed to read model\n");
		setlocale(LC_ALL, old_locale);
		free(old_locale);
		free(model->rho);
		free(model->label);
		free(model->nSV);
		free(model);
		return NULL;
	}

	// read sv_coef and SV

	int elements = 0;
	long pos = ftell(fp);

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	char *p,*endptr,*idx,*val;

	while(readline(fp)!=NULL)
	{
		p = strtok(line,":");
		while(1)
		{
			p = strtok(NULL,":");
			if(p == NULL)
				break;
			++elements;
		}
	}
	elements += model->l;

	fseek(fp,pos,SEEK_SET);

	int m = model->nr_class - 1;
	int l = model->l;
	model->sv_coef = Malloc(double *,m);
	int i;
	for(i=0;i<m;i++)
		model->sv_coef[i] = Malloc(double,l);
	model->SV = Malloc(svm_node*,l);
	svm_node *x_space = NULL;
	if(l>0) x_space = Malloc(svm_node,elements);

	int j=0;
	for(i=0;i<l;i++)
	{
		readline(fp);
		model->SV[i] = &x_space[j];

		p = strtok(line, " \t");
		model->sv_coef[0][i] = strtod(p,&endptr);
		for(int k=1;k<m;k++)
		{
			p = strtok(NULL, " \t");
			model->sv_coef[k][i] = strtod(p,&endptr);
		}

		while(1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if(val == NULL)
				break;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			x_space[j].value = strtod(val,&endptr);

			++j;
		}
		x_space[j++].index = -1;
	}
	free(line);

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0)
		return NULL;

	model->free_sv = 1;	// XXX
	return model;
}

void svm_free_model_content(svm_model* model_ptr)
{
	if(model_ptr->free_sv && model_ptr->l > 0 && model_ptr->SV != NULL)
		free((void *)(model_ptr->SV[0]));
	if(model_ptr->sv_coef)
	{
		for(int i=0;i<model_ptr->nr_class-1;i++)
			free(model_ptr->sv_coef[i]);
	}

	free(model_ptr->SV);
	model_ptr->SV = NULL;

	free(model_ptr->sv_coef);
	model_ptr->sv_coef = NULL;

	free(model_ptr->rho);
	model_ptr->rho = NULL;

	free(model_ptr->label);
	model_ptr->label = NULL;

	free(model_ptr->probA);
	model_ptr->probA = NULL;

	free(model_ptr->probB);
	model_ptr->probB = NULL;

	free(model_ptr->prob_density_marks);
	model_ptr->prob_density_marks = NULL;

	free(model_ptr->sv_indices);
	model_ptr->sv_indices = NULL;

	free(model_ptr->nSV);
	model_ptr->nSV = NULL;
}

void svm_free_and_destroy_model(svm_model** model_ptr_ptr)
{
	if(model_ptr_ptr != NULL && *model_ptr_ptr != NULL)
	{
		svm_free_model_content(*model_ptr_ptr);
		free(*model_ptr_ptr);
		*model_ptr_ptr = NULL;
	}
}

void svm_destroy_param(svm_parameter* param)
{
	free(param->weight_label);
	free(param->weight);
}

const char *svm_check_parameter(const svm_problem *prob, const svm_parameter *param)
{
	// svm_type

	int svm_type = param->svm_type;
	if(svm_type != C_SVC &&
	   svm_type != NU_SVC &&
	   svm_type != ONE_CLASS &&
	   svm_type != EPSILON_SVR &&
	   svm_type != NU_SVR && 
	   svm_type != W_SVM &&
	   svm_type != SVM_PLUS)
		return "unknown svm type";

	// kernel_type, degree

	int kernel_type = param->kernel_type;
	if(kernel_type != LINEAR &&
	   kernel_type != POLY &&
	   kernel_type != RBF &&
	   kernel_type != SIGMOID &&
	   kernel_type != PRECOMPUTED)
		return "unknown kernel type";

	if((kernel_type == POLY || kernel_type == RBF || kernel_type == SIGMOID) &&
	   param->gamma < 0)
		return "gamma < 0";

	if(kernel_type == POLY && param->degree < 0)
		return "degree of polynomial kernel < 0";

	// cache_size,eps,C,nu,p,shrinking

	if(param->cache_size <= 0)
		return "cache_size <= 0";

	if(param->eps <= 0)
		return "eps <= 0";

	if(svm_type == C_SVC ||
	   svm_type == SVM_PLUS ||
	   svm_type == EPSILON_SVR ||
	   svm_type == NU_SVR)
		if(param->C <= 0)
			return "C <= 0";

	if(svm_type == NU_SVC ||
	   svm_type == ONE_CLASS ||
	   svm_type == NU_SVR)
		if(param->nu <= 0 || param->nu > 1)
			return "nu <= 0 or nu > 1";

	if(svm_type == SVM_PLUS)
		if(param->tau <= 0)
			return "tau <= 0";


	if(svm_type == EPSILON_SVR)
		if(param->p < 0)
			return "p < 0";

	if(param->shrinking != 0 &&
	   param->shrinking != 1)
		return "shrinking != 0 and shrinking != 1";

	if(param->probability != 0 &&
	   param->probability != 1)
		return "probability != 0 and probability != 1";


	// check whether nu-svc is feasible

	if(svm_type == NU_SVC)
	{
		int l = prob->l;
		int max_nr_class = 16;
		int nr_class = 0;
		int *label = Malloc(int,max_nr_class);
		int *count = Malloc(int,max_nr_class);

		int i;
		for(i=0;i<l;i++)
		{
			int this_label = (int)prob->y[i];
			int j;
			for(j=0;j<nr_class;j++)
				if(this_label == label[j])
				{
					++count[j];
					break;
				}
			if(j == nr_class)
			{
				if(nr_class == max_nr_class)
				{
					max_nr_class *= 2;
					label = (int *)realloc(label,max_nr_class*sizeof(int));
					count = (int *)realloc(count,max_nr_class*sizeof(int));
				}
				label[nr_class] = this_label;
				count[nr_class] = 1;
				++nr_class;
			}
		}

		for(i=0;i<nr_class;i++)
		{
			int n1 = count[i];
			for(int j=i+1;j<nr_class;j++)
			{
				int n2 = count[j];
				if(param->nu*(n1+n2)/2 > min(n1,n2))
				{
					free(label);
					free(count);
					return "specified nu is infeasible";
				}
			}
		}
		free(label);
		free(count);
	}
	
	return NULL;
}

int svm_check_probability_model(const svm_model *model)
{
	return
		((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
		 model->probA!=NULL && model->probB!=NULL) ||
		(model->param.svm_type == ONE_CLASS && model->prob_density_marks!=NULL) ||
		((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
		 model->probA!=NULL);
}

void svm_set_print_string_function(void (*print_func)(const char *))
{
	if(print_func == NULL)
		svm_print_string = &print_string_stdout;
	else
		svm_print_string = print_func;
}
