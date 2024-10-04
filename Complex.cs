using System;
using UnityEngine;
public struct Complex
{
    public float real;
    public float imaginary;

    // Constructor to initialize the complex number
    public Complex(float r, float i)
    {
        real = r;
        imaginary = i;
    }

    // Method to get the magnitude squared (|ψ|^2)
    public float MagnitudeSquared()
    {
        return real * real + imaginary * imaginary;
    }

    // Linear interpolation between two complex numbers
    public static Complex Lerp(Complex a, Complex b, float t)
    {
        float newReal = Mathf.Lerp(a.real, b.real, t);
        float newImaginary = Mathf.Lerp(a.imaginary, b.imaginary, t);
        return new Complex(newReal, newImaginary);
    }

    // Overriding ToString for better readability
    public override string ToString()
    {
        return string.Format("({0} + {1}i)", real, imaginary);
    }

    // Addition of two complex numbers
    public static Complex operator +(Complex a, Complex b)
    {
        return new Complex(a.real + b.real, a.imaginary + b.imaginary);
    }

    // Multiplication of two complex numbers
    public static Complex operator *(Complex a, Complex b)
    {
        return new Complex(a.real * b.real - a.imaginary * b.imaginary, a.real * b.imaginary + a.imaginary * b.real);
    }
}
