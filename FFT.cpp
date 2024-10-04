#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

class FFT
{
public:
    using Complex = std::complex<double>;
    using ComplexArray = std::vector<Complex>;

    // Compute the FFT of the input array
    static ComplexArray Compute(const ComplexArray& input)
    {
        int n = input.size();
        if (n <= 1) return input;

        // Divide
        ComplexArray evenTerms(n / 2);
        ComplexArray oddTerms(n / 2);
        for (int i = 0; i < n / 2; ++i)
        {
            evenTerms[i] = input[i * 2];
            oddTerms[i] = input[i * 2 + 1];
        }

        // Conquer
        ComplexArray fftEven = Compute(evenTerms);
        ComplexArray fftOdd = Compute(oddTerms);

        // Combine
        ComplexArray result(n);
        for (int k = 0; k < n / 2; ++k)
        {
            Complex t = std::polar(1.0, -2 * M_PI * k / n) * fftOdd[k];
            result[k] = fftEven[k] + t;
            result[k + n / 2] = fftEven[k] - t;
        }

        return result;
    }

    // Compute the Inverse FFT
    static ComplexArray Inverse(const ComplexArray& input)
    {
        int n = input.size();
        ComplexArray conjugatedInput(n);

        // Take the conjugate of the input
        for (int i = 0; i < n; ++i)
        {
            conjugatedInput[i] = std::conj(input[i]);
        }

        // Apply FFT
        ComplexArray fftResult = Compute(conjugatedInput);

        // Take the conjugate again and scale by 1/n
        for (int i = 0; i < n; ++i)
        {
            fftResult[i] = std::conj(fftResult[i]) / static_cast<double>(n);
        }

        return fftResult;
    }
};
