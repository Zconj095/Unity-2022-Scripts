using System.Collections.Generic;
using UnityEngine;

public class PeriodicTableGenerator : MonoBehaviour
{
    [Header("Periodic Table of Elements")]
    public List<Element> elements = new List<Element>(); // List to store all elements

    void Start()
    {
        GeneratePeriodicTable();
    }

    [ContextMenu("Generate Periodic Table")]
    public void GeneratePeriodicTable()
    {
        elements.Clear(); // Clear the list before generating

        // elements (atomic number, name, symbol, atomic mass)
        elements.Add(new Element(1, "Hydrogen", "H", 1.008f));
        elements.Add(new Element(2, "Helium", "He", 4.0026f));
        elements.Add(new Element(3, "Lithium", "Li", 6.94f));
        elements.Add(new Element(4, "Beryllium", "Be", 9.0122f));
        elements.Add(new Element(5, "Boron", "B", 10.81f));
        elements.Add(new Element(6, "Carbon", "C", 12.011f));
        elements.Add(new Element(7, "Nitrogen", "N", 14.007f));
        elements.Add(new Element(8, "Oxygen", "O", 15.999f));
        elements.Add(new Element(9, "Fluorine", "F", 18.998f));
        elements.Add(new Element(10, "Neon", "Ne", 20.180f));
        elements.Add(new Element(11, "Sodium", "Na", 22.990f));
        elements.Add(new Element(12, "Magnesium", "Mg", 24.305f));
        elements.Add(new Element(13, "Aluminum", "Al", 26.982f));
        elements.Add(new Element(14, "Silicon", "Si", 28.085f));
        elements.Add(new Element(15, "Phosphorus", "P", 30.974f));
        elements.Add(new Element(16, "Sulfur", "S", 32.06f));
        elements.Add(new Element(17, "Chlorine", "Cl", 35.45f));
        elements.Add(new Element(18, "Argon", "Ar", 39.948f));
        elements.Add(new Element(19, "Potassium", "K", 39.098f));
        elements.Add(new Element(20, "Calcium", "Ca", 40.078f));
        elements.Add(new Element(21, "Scandium", "Sc", 44.956f));
        elements.Add(new Element(22, "Titanium", "Ti", 47.867f));
        elements.Add(new Element(23, "Vanadium", "V", 50.942f));
        elements.Add(new Element(24, "Chromium", "Cr", 51.996f));
        elements.Add(new Element(25, "Manganese", "Mn", 54.938f));
        elements.Add(new Element(26, "Iron", "Fe", 55.845f));
        elements.Add(new Element(27, "Cobalt", "Co", 58.933f));
        elements.Add(new Element(28, "Nickel", "Ni", 58.693f));
        elements.Add(new Element(29, "Copper", "Cu", 63.546f));
        elements.Add(new Element(30, "Zinc", "Zn", 65.38f));
        elements.Add(new Element(31, "Gallium", "Ga", 69.723f));
        elements.Add(new Element(32, "Germanium", "Ge", 72.63f));
        elements.Add(new Element(33, "Arsenic", "As", 74.922f));
        elements.Add(new Element(34, "Selenium", "Se", 78.971f));
        elements.Add(new Element(35, "Bromine", "Br", 79.904f));
        elements.Add(new Element(36, "Krypton", "Kr", 83.798f));
        elements.Add(new Element(37, "Rubidium", "Rb", 85.468f));
        elements.Add(new Element(38, "Strontium", "Sr", 87.62f));
        elements.Add(new Element(39, "Yttrium", "Y", 88.906f));
        elements.Add(new Element(40, "Zirconium", "Zr", 91.224f));
        elements.Add(new Element(41, "Niobium", "Nb", 92.906f));
        elements.Add(new Element(42, "Molybdenum", "Mo", 95.95f));
        elements.Add(new Element(43, "Technetium", "Tc", 98.0f)); // Radioactive, no stable isotopes
        elements.Add(new Element(44, "Ruthenium", "Ru", 101.07f));
        elements.Add(new Element(45, "Rhodium", "Rh", 102.91f));
        elements.Add(new Element(46, "Palladium", "Pd", 106.42f));
        elements.Add(new Element(47, "Silver", "Ag", 107.87f));
        elements.Add(new Element(48, "Cadmium", "Cd", 112.41f));
        elements.Add(new Element(49, "Indium", "In", 114.82f));
        elements.Add(new Element(50, "Tin", "Sn", 118.71f));
        elements.Add(new Element(51, "Antimony", "Sb", 121.76f));
        elements.Add(new Element(52, "Tellurium", "Te", 127.6f));
        elements.Add(new Element(53, "Iodine", "I", 126.90f));
        elements.Add(new Element(54, "Xenon", "Xe", 131.29f));
        elements.Add(new Element(55, "Cesium", "Cs", 132.91f));
        elements.Add(new Element(56, "Barium", "Ba", 137.33f));
        elements.Add(new Element(57, "Lanthanum", "La", 138.91f));
        elements.Add(new Element(58, "Cerium", "Ce", 140.12f));
        elements.Add(new Element(59, "Praseodymium", "Pr", 140.91f));
        elements.Add(new Element(60, "Neodymium", "Nd", 144.24f));
        elements.Add(new Element(61, "Promethium", "Pm", 145.0f)); // Radioactive, no stable isotopes
        elements.Add(new Element(62, "Samarium", "Sm", 150.36f));
        elements.Add(new Element(63, "Europium", "Eu", 151.96f));
        elements.Add(new Element(64, "Gadolinium", "Gd", 157.25f));
        elements.Add(new Element(65, "Terbium", "Tb", 158.93f));
        elements.Add(new Element(66, "Dysprosium", "Dy", 162.50f));
        elements.Add(new Element(67, "Holmium", "Ho", 164.93f));
        elements.Add(new Element(68, "Erbium", "Er", 167.26f));
        elements.Add(new Element(69, "Thulium", "Tm", 168.93f));
        elements.Add(new Element(70, "Ytterbium", "Yb", 173.04f));
        elements.Add(new Element(71, "Lutetium", "Lu", 174.97f));
        elements.Add(new Element(72, "Hafnium", "Hf", 178.49f));
        elements.Add(new Element(73, "Tantalum", "Ta", 180.95f));
        elements.Add(new Element(74, "Tungsten", "W", 183.84f));
        elements.Add(new Element(75, "Rhenium", "Re", 186.21f));
        elements.Add(new Element(76, "Osmium", "Os", 190.23f));
        elements.Add(new Element(77, "Iridium", "Ir", 192.22f));
        elements.Add(new Element(78, "Platinum", "Pt", 195.08f));
        elements.Add(new Element(79, "Gold", "Au", 196.97f));
        elements.Add(new Element(80, "Mercury", "Hg", 200.59f));
        elements.Add(new Element(81, "Thallium", "Tl", 204.38f));
        elements.Add(new Element(82, "Lead", "Pb", 207.2f));
        elements.Add(new Element(83, "Bismuth", "Bi", 208.98f));
        elements.Add(new Element(84, "Polonium", "Po", 209.0f)); // Radioactive
        elements.Add(new Element(85, "Astatine", "At", 210.0f)); // Radioactive
        elements.Add(new Element(86, "Radon", "Rn", 222.0f)); // Radioactive
        elements.Add(new Element(87, "Francium", "Fr", 223.0f)); // Radioactive
        elements.Add(new Element(88, "Radium", "Ra", 226.0f)); // Radioactive
        elements.Add(new Element(89, "Actinium", "Ac", 227.0f)); // Radioactive
        elements.Add(new Element(90, "Thorium", "Th", 232.04f));
        elements.Add(new Element(91, "Protactinium", "Pa", 231.04f));
        elements.Add(new Element(92, "Uranium", "U", 238.03f));
        elements.Add(new Element(93, "Neptunium", "Np", 237.0f)); // Radioactive
        elements.Add(new Element(94, "Plutonium", "Pu", 244.0f)); // Radioactive
        elements.Add(new Element(95, "Americium", "Am", 243.0f)); // Radioactive
        elements.Add(new Element(96, "Curium", "Cm", 247.0f)); // Radioactive
        elements.Add(new Element(97, "Berkelium", "Bk", 247.0f)); // Radioactive
        elements.Add(new Element(98, "Californium", "Cf", 251.0f)); // Radioactive
        elements.Add(new Element(99, "Einsteinium", "Es", 252.0f)); // Radioactive
        elements.Add(new Element(100, "Fermium", "Fm", 257.0f)); // Radioactive
        elements.Add(new Element(101, "Mendelevium", "Md", 258.0f)); // Radioactive
        elements.Add(new Element(102, "Nobelium", "No", 259.0f)); // Radioactive
        elements.Add(new Element(103, "Lawrencium", "Lr", 262.0f)); // Radioactive
        elements.Add(new Element(104, "Rutherfordium", "Rf", 267.0f)); // Radioactive
        elements.Add(new Element(105, "Dubnium", "Db", 270.0f)); // Radioactive
        elements.Add(new Element(106, "Seaborgium", "Sg", 271.0f)); // Radioactive
        elements.Add(new Element(107, "Bohrium", "Bh", 270.0f)); // Radioactive
        elements.Add(new Element(108, "Hassium", "Hs", 277.0f)); // Radioactive
        elements.Add(new Element(109, "Meitnerium", "Mt", 278.0f)); // Radioactive
        elements.Add(new Element(110, "Darmstadtium", "Ds", 281.0f)); // Radioactive
        elements.Add(new Element(111, "Roentgenium", "Rg", 282.0f)); // Radioactive
        elements.Add(new Element(112, "Copernicium", "Cn", 285.0f)); // Radioactive
        elements.Add(new Element(113, "Nihonium", "Nh", 286.0f)); // Radioactive
        elements.Add(new Element(114, "Flerovium", "Fl", 289.0f)); // Radioactive
        elements.Add(new Element(115, "Moscovium", "Mc", 290.0f)); // Radioactive
        elements.Add(new Element(116, "Livermorium", "Lv", 293.0f)); // Radioactive
        elements.Add(new Element(117, "Tennessine", "Ts", 294.0f)); // Radioactive
        elements.Add(new Element(118, "Oganesson", "Og", 294.0f)); // Radioactive

    }
}
