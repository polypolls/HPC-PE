// add random number generator here

#ifndef MLPMath_H
#define MLPMath_H

#include <cstdlib>
#include <vector>
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <random>
#include <algorithm>

#include "p4_f32vec4.h"

class MLPMath {
private:

public:

	/**
	* @brief MatMul2D1D multiplies 2D with 1D matrix with commensurate dimensions.
	* Checks if input matrices are compatible.
	* @param A(2D) and B(1D), the matrices to be multiplied.
	* @returns A 1D vector C = A x B
	*/
	static std::vector<float> MatMul2D1D(std::vector<std::vector<float>>& A, std::vector<float>& B);

	static std::vector<fvec> MatMul2D1D(std::vector<std::vector<float>>& A, std::vector<fvec>& B);


	static std::vector<float> MatMul1D1D(std::vector<float>& A, std::vector<float>& B);

	/**
	 * @brief random_weight returns a random value based on multiple modes
	 * @param mode sets the mode the random value calculation is based on
	 * @param layerSize is the size of layer X used for random calculation
	 * @param nextLayerSize is the size of layer X+1 used for random calculation
	 * @return returns a double random value based on given input parameters
	 */
	static float CalculateRandomWeight( int layerSize, int nextLayerSize);


	/**
	* @brief Applies softmax to input.
	* Uses maxvalue for numerical stability
	* @param A 1D vector input
	* @returns A 1D vector
	*/
	static std::vector<float> applySoftmax(std::vector<float>& input);

	static std::vector<fvec> applySoftmax(std::vector<fvec>& input);


	/**
	* @brief Transposes 2D matrix
	* @param A 2D vector
	* @returns A 2D vector
	*/
	static std::vector<std::vector<float>> Transpose2D(std::vector<std::vector<float>>& A);

	static std::vector<float> affineTransform(std::vector<std::vector<float>>& weight, std::vector<float>& input, std::vector<float>& bias);

	static std::vector<fvec> affineTransform(std::vector<std::vector<float>>& weight, std::vector<fvec>& input, std::vector<float>& bias);


	/**
	 * @brief Applies sigmoid elementwise. Avoids overflow.
	 * @param 1D vector
	 * @return returns 1D vector
	 */
	static std::vector<float> sigmoid(std::vector<float>& input);

	/**
	 * @brief Adds vector elementwise.
	 * @param 1D vector, 1D vector
	 * @return returns 1D vector
	 */
	static std::vector<float> addVector(const std::vector<float>& A, const std::vector<float>& B);

	static std::vector<float> addVector(const std::vector<float>& A, const std::vector<fvec>& B);


	/**
	 * @brief Applies tanh elementwise to input array
	 * @param 1D vector
	 * @return returns 1D vector
	 */
	static std::vector<float> applyTanH(std::vector<float>& rawHidden);

	static std::vector<fvec> applyTanH(std::vector<fvec>& rawHidden);

	static std::vector<fvec> applyReLU(std::vector<fvec>& rawHidden);


	static std::vector<std::vector<float>> addMatrix2D(std::vector<std::vector<float>>& A, std::vector<std::vector<float>>& B);

};

#endif // MLPMath_H
