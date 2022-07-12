/* Please see the "LICENSE" file for the copyright information. */

/*
Notice: This SpaSNE software is adapted from the bhtsne code 
       (github.com/lvdmaaten/bhtsne). 
       Please see the "LICENSE" file for copyright details of the bhtsne 
       software. The implementation of the bhtsne software is described in the 
       publication "Accelerating t-SNE using Tree-Based Algorithms" 
       (https://jmlr.org/papers/v15/vandermaaten14a.html). 
*/

#ifndef SPTREE_H
#define SPTREE_H

class Cell {

    unsigned int dimension;
    double* corner;
    double* width;
    
    
public:
    Cell(unsigned int inp_dimension);
    Cell(unsigned int inp_dimension, double* inp_corner, double* inp_width);
    ~Cell();
    
    double getCorner(unsigned int d);
    double getWidth(unsigned int d);
    void setCorner(unsigned int d, double val);
    void setWidth(unsigned int d, double val);
    bool containsPoint(double point[]);
};


class SPTree
{
    
    // Fixed constants
    static const unsigned int QT_NODE_CAPACITY = 1;

    // A buffer we use when doing force computations
    double* buff;
    
    // Properties of this node in the tree
    SPTree* parent;
    unsigned int dimension;
    bool is_leaf;
    unsigned int size;
    unsigned int cum_size;
        
    // Axis-aligned bounding box stored as a center with half-dimensions to represent the boundaries of this quad tree
    Cell* boundary;
    
    // Indices in this space-partitioning tree node, corresponding center-of-mass, and list of all children
    double* data;
    double* center_of_mass;
    unsigned int index[QT_NODE_CAPACITY];
    
    // Children
    SPTree** children;
    unsigned int no_children;
    
public:
    SPTree(unsigned int D, double* inp_data, unsigned int N);
    SPTree(unsigned int D, double* inp_data, double* inp_corner, double* inp_width);
    SPTree(unsigned int D, double* inp_data, unsigned int N, double* inp_corner, double* inp_width);
    SPTree(SPTree* inp_parent, unsigned int D, double* inp_data, unsigned int N, double* inp_corner, double* inp_width);
    SPTree(SPTree* inp_parent, unsigned int D, double* inp_data, double* inp_corner, double* inp_width);
    ~SPTree();
    void setData(double* inp_data);
    SPTree* getParent();
    void construct(Cell boundary);
    bool insert(unsigned int new_index);
    void subdivide();
    bool isCorrect();
    void rebuildTree();
    void getAllIndices(unsigned int* indices);
    unsigned int getDepth();
    void computeNonEdgeForces(unsigned int point_index, double theta, double neg_f[], double* sum_Q);
    void computeNonEdgeForces_SpaSNE(unsigned int point_index, double theta, double neg_f[], double* sum_Q, double neg_f_global[], double* sum_Q_global);
    void computeEdgeForces(unsigned int* row_P, unsigned int* col_P, double* val_P, int N, double* pos_f);
//    void computeEdgeForces_SpaSNE(unsigned int* row_P, unsigned int* col_P, double* val_P, int N, double* pos_f, double* P_global, double* P_space, double alpha, double beta); 
	void computeEdgeForces_SpaSNE(int N, double* pos_f, double* P, double* P_global, double* P_space, double alpha, double beta);
    void print();
    
private:
    void init(SPTree* inp_parent, unsigned int D, double* inp_data, double* inp_corner, double* inp_width);
    void fill(unsigned int N);
    unsigned int getAllIndices(unsigned int* indices, unsigned int loc);
    bool isChild(unsigned int test_index, unsigned int start, unsigned int end);
};

#endif
