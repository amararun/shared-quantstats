# QuantStats Application Styling Guide

This document outlines the styling elements used in the QuantStats application for consistent replication in other applications.

## Color Palette

### Indigo Theme Colors
```css
--indigo-950: #1e1b4b;  /* Very dark indigo - Header gradient start */
--indigo-900: #312e81;  /* Dark indigo - Header gradient end */
--indigo-800: #3730a3;  /* Medium-dark indigo - Badge backgrounds */
--indigo-700: #4338ca;  /* Medium indigo - Button hover states, borders */
--indigo-600: #4f46e5;  /* Primary button color */
--indigo-200: #c7d2fe;  /* Light indigo - Subtle accents */
--indigo-100: #e0e7ff;  /* Very light indigo - Footer border */
--indigo-50: #eef2ff;   /* Nearly white indigo - Subtle backgrounds */
```

### Other Colors
- Background color: `#f9f9f9` (light gray) or `rgb(248 250 252)` (slate-50)
- Text color: `#333` (dark gray)
- Card background: `white`
- Card border: `rgb(229, 231, 235)` (gray-200)
- Success/View Report Button: `bg-green-600` (hover: `bg-green-700`)
- Error message background: `bg-red-50`
- Error message text: `text-red-700`
- Error message border: `border-red-200`
- Disclaimer background: `bg-amber-50`
- Disclaimer text: `text-amber-800`
- Disclaimer border: `border-amber-200`

## Typography

### Font Families
```css
font-family: 'Roboto', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
```

### Font Sizes
- Header title: `1.25rem` (text-xl)
- Header subtitle: `1rem` (text-base)
- Card headings: `1.25rem` (text-xl)
- Form labels: `0.875rem` (text-sm) with `font-medium`
- Input text: `0.875rem` (text-sm)
- Button text: `0.875rem` (text-sm) with `font-medium`
- Helper text: `0.75rem` (text-xs) with `text-gray-500`
- Footer text: `0.75rem` (text-xs)

## Components

### Header
- Gradient background: `linear-gradient(135deg, var(--indigo-950) 0%, var(--indigo-900) 100%)`
- Box shadow: `0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)`
- Border-bottom: `1px solid rgba(255, 255, 255, 0.05)`

### Cards
- Background: `white`
- Border-radius: `0.75rem` (rounded-xl)
- Box-shadow: `shadow-sm` (Tailwind) or `0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)`
- Border: `1px solid rgb(229, 231, 235)` (border-gray-200)
- Padding: `1rem` (p-4)

### Form Elements
- Label: `block text-sm font-medium text-gray-700`
- Input fields: 
  - `w-full px-3 py-1.5 text-sm border border-gray-300 rounded-md`
  - Focus state: `focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500`
- Button (primary): 
  - `px-4 py-1.5 text-sm bg-indigo-600 text-white font-medium rounded-md shadow-sm`
  - Hover state: `hover:bg-indigo-700 transition-colors`
- Button (success/view report):
  - `px-3 py-1.5 text-sm bg-green-600 text-white font-medium rounded-md shadow-sm`
  - Hover state: `hover:bg-green-700 transition-colors`

### Footer
- Background: `bg-white/50` (semi-transparent white)
- Border-top: `border-t border-indigo-100`
- Text color: `text-indigo-950/70` (semi-transparent dark indigo)
- Links: `text-indigo-600` (hover: `text-indigo-700 hover:underline`)

### Loading Spinner
```css
.spinner {
  width: 20px;
  height: 20px;
  border: 2px solid #e5e7eb;
  border-top: 2px solid #4f46e5;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
```

## Layout

- Maximum content width: `max-w-7xl` (1280px)
- Main form layout: `grid grid-cols-3 gap-4` for inputs
- Responsive date fields: `grid grid-cols-2 gap-4`
- About section layout: `grid grid-cols-2 gap-2`
- Footer layout: `flex flex-col md:flex-row justify-between items-center gap-1`

## Responsive Design

- Mobile-first approach
- Breakpoint: `768px` (md in Tailwind)
- Footer transitions from column to row layout at this breakpoint
- Header adjusts spacing and alignment for mobile

## Framework
- Uses Tailwind CSS (CDN): `<script src="https://cdn.tailwindcss.com"></script>`
- Font Awesome icons: `<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">` 