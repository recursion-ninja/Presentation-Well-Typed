{- |


= Needleman–Wunsch String Alignment


== Presentation Overview

  1. Understand the /problem/.

  2. Understand the /code/.


You can follow along by accessing the module at the following
[publicly available GitHub repository.](https://github.com/recursion-ninja/Presentation-Well-Typed)

__Ask questions any time!__


== Algorithm Summary


The module contains the code for the Needleman–Wunsch string alignment algorithm.
This is a dynamic programming algorithm designed to find a global minimum distance.

The algorithm was originally published in the 1970 paper,
/"A general method applicable to the search for similarities in the amino acid sequence of two proteins."/
[The original paper is accessible here](https://doi.org/10.1016%2F0022-2836%2870%2990057-4)


The Needleman–Wunsch algorithm operates on three input parameters:

  * Σ: A finite alphabet of symbols, such that, @-@ ∈ Σ

  * σ: A distance measure between symbols of Σ, such that, σ: Σ×Σ ↦ ℕ

  * 𝙨₁, 𝙨₂: Two strings of symbols, such that, |𝙨₁| = 𝙢, |𝙨₂| = 𝙣, 𝙢 ≤ 𝙣, and 𝙨₁, 𝙨₂ ∈ Σ✲


The /complete/ algorithm Needleman–Wunsch produces two outputs:

  * An alignment of strings 𝙨₁ and 𝙨₂ which maximizes similarity

  * The minimal distance between strings 𝙨₁ and 𝙨₂, as defined by σ


However, /for brevity of the presentation,/ we will be exploring code which /only computes the alignment distance./


== Algorithm Description

It is easiest to understand the algorithm by using a running example as the alogrithm's operation is described.
Let us define the input parameters as the following:

  * @
    Σ = { -, A, C, G, T }
    @

  * @
    σ(x,y) = 0 iff x = y
    σ(x,y) = 2 iff x ≠ y and either x = ― or y = ―
    σ(x,y) = 1 otherwise
    @

  * @
    𝙨₁ = GATTACA
    𝙨₂ = ATTAGAGACA
    𝙢  = 7
    𝙣  = 10
    @

__Example Alignment__

@
Needleman–Wunsch( σ, GATTACA, ATTAGAGACA )

𝙨₁ = GATT--A--CA
𝙨₂ = -ATTAGAGACA
@


The algorithm aligns the strings by constructing an (𝙢 + 1) × (𝙣 + 1) matrix 𝙈.
Each both 𝙨₁ and 𝙨₂ have a ― symbol prepended to them.
The smaller string 𝙨₁ is placed before the rows of 𝙈
The longer string 𝙨₂ is placed above the columns of 𝙈
The initial configuration of 𝙈

    +---+---+---+---+---+---+---+---+---+---+---+---+
    | × | - | A | T | T | A | G | A | G | A | C | A |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | - |   |   |   |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | G |   |   |   |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | A |   |   |   |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | T |   |   |   |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | T |   |   |   |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | A |   |   |   |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | C |   |   |   |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | A |   |   |   |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+


First, the algorithm initializes the cell 𝙈₀₀ with a distance of 0:

    +---+---+---+---+---+---+---+---+---+---+---+---+
    | × | - | A | T | T | A | G | A | G | A | C | A |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | - | 0 |   |   |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | G |   |   |   |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | A |   |   |   |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | T |   |   |   |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | T |   |   |   |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | A |   |   |   |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | C |   |   |   |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | A |   |   |   |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+


Second, the algorithm applies a memoized lookup function to calculate the value of 𝙈[i,j]:

@
   memo: ℤ×ℤ ↦ ℕ ∪ { ∞ }
   memo(i,j) = ∞ if i < 0
   memo(i,j) = ∞ if j < 0
   memo(i,j) = otherwise:
        minimum
            [ 𝙈[i  , j-1] + σ(  ―, 𝙨₂[j])
            , 𝙈[i-1, j-1] + σ(𝙨₁[i], 𝙨₂[j])
            , 𝙈[i-1, j  ] + σ(𝙨₁[i],   ―)
            ]
@

The final cell 𝙈ₘₙ of the matrix 𝙈 contains the distance between 𝙨₁ and 𝙨₂,
measured by σ in the "alignment space."

Each cell in 𝙈, except for cell 𝙈₀₀, depends on the value of the cell direct above it,
directly to it's left, as well as the cell diagonally above and to it's left.
However, cells in the first row should not consider the cells above them.
Similarly, cells in the the first column should not consider the cells to their left.

/Note:/
The function @memo@ takes possibly negative integer indices.
Giving @memo@ this domain greatly simplifies the memoiztion definition,
since we no longer need to special case cells in the first row or column.

Knowing that each cell in 𝙈 (except 𝙈₀₀) is defined by the row above it and
the cell to it's left, the algorithm generates 𝙈 row-by-row, from top to bottom.


Consider the cell 𝙈[0,1] in the first row:

@
    𝙈[0,1] = memo(0,1)
           = minimum [ memo(0,0) + σ(―, A), memo(-1,0) + σ(―, A), memo(-1,1) + σ(―, ―) ]
           = minimum [ 0 + 2, ∞ + 2, ∞ + 0 ]
           = minimum [ 2, ∞, ∞ ]
           = 2
@

Here is the result of generating the rest of the first row:

    +---+---+---+---+---+---+---+---+---+---+---+---+
    | × | - | A | T | T | A | G | A | G | A | C | A |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | - | 0 | 2 | 4 | 6 | 8 | 10| 12| 14| 16| 18| 20|
    +---+---+---+---+---+---+---+---+---+---+---+---+


Consider the cell 𝙈[1,1] in the second row:

@
    𝙈[0,1] = memo(1,1)
           = minimum [ memo(1,0) + σ(―, @G@), memo(0,0) + σ(@G@, A), memo(0,1) + σ(A, ―) ]
           = minimum [ 2 + 2, 0 + 1, 2 + 2 ]
           = minimum [ 4, 1, 4 ]
           = 1
@

Here is the result of generating the entire second row:

    +---+---+---+---+---+---+---+---+---+---+---+---+
    | × | - | A | T | T | A | G | A | G | A | C | A |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | - | 0 | 2 | 4 | 6 | 8 | 10| 12| 14| 16| 18| 20|
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | G | 2 | 1 | 3 | 5 | 6 | 8 | 10| 12| 14| 16| 18|
    +---+---+---+---+---+---+---+---+---+---+---+---+

As the algorithm proceeds row-by-row, the entirety of the matrix is created:

    +---+---+---+---+---+---+---+---+---+---+---+---+
    | × | - | A | T | T | A | G | A | G | A | C | A |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | - | 0 | 2 | 4 | 6 | 8 | 10| 12| 14| 16| 18| 20|
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | G | 2 | 1 | 3 | 5 | 6 | 8 | 10| 12| 14| 16| 18|
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | A | 4 | 2 | 2 | 4 | 5 | 7 | 8 | 10| 12| 14| 16|
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | T | 6 | 4 | 2 | 2 | 4 | 6 | 8 | 9 | 11| 13| 15|
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | T | 8 | 6 | 4 | 2 | 3 | 5 | 7 | 8 | 10| 12| 14|
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | A | 10| 8 | 6 | 4 | 2 | 4 | 5 | 7 | 9 | 11| 12|
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | C | 12| 10| 8 | 6 | 4 | 3 | 5 | 6 | 8 | 9 | 11|
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | A | 14| 12| 10| 8 | 6 | 5 | 3 | 5 | 6 | 8 | 9 |
    +---+---+---+---+---+---+---+---+---+---+---+---+


If one were to calculate more than just the alignment distance,
and also compute the /actual/ alignment of 𝙨₁ and 𝙨₂,
this can be achieved by "tracing backwards" from the final cell 𝙈ₘₙ
to the first cell 𝙈₀₀.

Beginning at 𝙈ₘₙ choose to move in one of three directions, either ←, ↖, or ↑;
selecting the neighboring cell with the lowest distance value.
Repeat this process until arriving at 𝙈₀₀:

  * For each ←, align @  ―  @ with @𝙨₂[j]@.
  * For each ↖, align @𝙨₁[i]@ with @𝙨₂[j]@.
  * For each ↖, align @𝙨₁[i]@ with @  ―  @.

One possible "traceback" path of the matrix 𝙈 is depicited below:

    +---+---+---+---+---+---+---+---+---+---+---+---+
    | × | - | A | T | T | A | G | A | G | A | C | A |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | - | ↖ |   |   |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | G |   | ↖ |   |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | A |   | ↑ |   |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | T |   |   | ↖ |   |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | T |   |   |   | ↖ |   |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | A |   |   |   |   | ↖ |   |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | C |   |   |   |   |   | ↖ |   |   |   |   |   |
    +---+---+---+---+---+---+---+---+---+---+---+---+
    | A |   |   |   |   |   |   | ↖ | ← | ← | ← | ← |
    +---+---+---+---+---+---+---+---+---+---+---+---+

This "traceback" corresponds to the following string alignment:
@
  𝙨₁ = GATTACA----
  𝙨₂ = -ATTAGAGACA
@

== Asymptotic Analysis

Because the Needleman–Wunsch algorithm, as originally described, fully populates
an (𝙢 + 1) × (𝙣 + 1) matrix, it has the following space and time complexities:

  • 𝓞(𝙢 * 𝙣): Worst-case space complexity

  • 𝓞(𝙢 * 𝙣): Worst-case time complexity


However, there have been numerous improvements made on the original 1970s
description of the algorithm. The implementation in this module utilizes the fact
that the matrix 𝙈 can be comupted row-by-row (or column-by-column), and only the
previous row needs to be retained to compute the current row. By retaining only
two rows of 𝙈 at a time, the algorithm benefits from an asymtotic improvement
in space complexity:

  • Θ(2 * 𝙢): Worst-case space complexity (improved)


Surprisingly, this improved version of the Needleman–Wunsch algorithm is
incredibly amenable to a functional programming style.

-}
module StringAlignment (
  module StringAlignment,
) where


import Data.List qualified as L
import Data.List.NonEmpty qualified as NE
import Data.List.NonEmpty (NonEmpty(..), (<|))
import Data.Word
-- Hide unsafe 'List' functions in order to use the safe versions from 'NonEmpty'
import Prelude hiding (head, last, repeat, scanl, tail, zip3)
-- These are needed for implementing type-class instances of 'Distance'
import Data.Bits
import Text.ParserCombinators.ReadPrec (get, pfail, (<++))
import Text.Read (Read (..))


{- |
'Distance' abstractly represents values of the domain ℕ ∪ { ∞ }.

The values of the data-type encode the following:

  * One value representing infinity
  * The remaining values represent a contiguous subset of ℕ which contains 0.

/Note:/
Concretely, 'Distance' is encoded using a 'Word16' for compactness reasons.
The 'maxBound' of the newtype is treated as the ∞ value.
To account for one less finite value representablle by 'Distance',
the 'Num' instance performs arithmetic operations modulo (2^16 - 1).
This can easily be changed to 'Word32' or 'Word64'.
-}
newtype Distance = D Word16
    deriving newtype (Bounded, Eq, Ord)
-- Instance derivations for (Enum, Num, Read, Show) moved to bottom of module for brevity.


{- |
The infinite 'Distance' value.
-}
infinity ∷ Distance
infinity = D maxBound


{- |
Define a measure to be used as σ.

Without loss of generality, select the same measure defined in the module header's example(s).
-}
measure :: Char -> Char -> Distance
measure x y | x == y = 0
measure '-' _ = 2
measure _ '-' = 2
measure _  _  = 1


{- |
Compute the align distance between the two strings under the given measure.
-}
needlemanWunschDefinition :: (Char -> Char -> Distance) -> String -> String -> Distance
needlemanWunschDefinition _ [] [] = 0 -- Do not operate on two empty strings
needlemanWunschDefinition σ in1 in2 =
    let -- | Conditionally swap input to ensure s1 ≤ s2
        (s1, s2)
          | length in1 <= length in2 = (in1, in2)
          | otherwise = (in2, in1)

        {- |
        Compute the value of the desired cell in the matrix 𝙈 using the distance measure σ.
        The resulting value of the desired cell is the minimum partial distance from 𝙈ᵢⱼ to 𝙈₀₀.
        This is calculated from the following memoized values:

            * The distance measure σ
            * The symbol from 𝙨₁ left of the desired row
            * The partial distance values in cells left of, diagonal to, and above the desired cell
            * The symbol from 𝙨₂ above of the desired column
        -}
        minimizeCellDistance :: Char -> Distance -> (Distance, Distance, Char) -> Distance
        minimizeCellDistance charL cell (cellD, cellA, charA) = minimum
            [ cell + σ  '-'  charA  
            , cellD + σ charL charA 
            , cellA + σ charL  '-'  
            ]

        {- |
        Generate the first row, which will be used as the accumulator seed for the subsequent rows.

        The first cell of the first row is set to 0.
        Any cell after the first is computed by using the value of the cell on it's left.

            +---+---+---+---+---+---+---+---+---+---+---+---+
            | × | - | A | T | T | A | G | A | G | A | C | A |
            +---+---+---+---+---+---+---+---+---+---+---+---+
            | - | 0 | 2 | 4 | 6 | 8 | 10| 12| 14| 16| 18| 20|
            +---+---+---+---+---+---+---+---+---+---+---+---+
            | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ |
            +---+---+---+---+---+---+---+---+---+---+---+---+

        -}
        firstRow :: NonEmpty Distance
        firstRow = NE.fromList . L.scanl (minimizeCellDistance '-') 0 $ L.zip3 (L.repeat infinity) (L.repeat infinity) s2

        -- | Compute a row of the memoization matrix 𝙈
        computeRow
          :: NonEmpty Distance -- ^ The previous row in 𝙈
          -> Char       -- ^ The symbol in 𝙨₁ which was placed in front of this row
          -> NonEmpty Distance -- ^ The current row of 𝙈
        computeRow prevRow charL =
            let {- |
                
                -}
                diagRow :: NonEmpty Distance
                diagRow = infinity <| prevRow

                {- |
                The result of the @scanl@ will include /at least two elements/,
                    1. The initial accumulator @infinity@.
                    2. The result of @minimizeCellDistance@ applied to the prepended '-' symbol.
                Hence the call to @fromList@ is safe after discarding the undesired @infinity@ value.
                -}
                finalize :: NonEmpty Distance -> NonEmpty Distance
                finalize = NE.fromList . NE.tail
                
            in  finalize . NE.scanl (minimizeCellDistance charL) infinity . zip3 diagRow prevRow $ '-' :| s2

    in  NE.last $ L.foldl' computeRow firstRow s1


instance Enum Distance where
    toEnum i = case i `compare` fromEnum (maxBound ∷ Word16) of
        LT → D $ toEnum i
        _ → maxBound


    fromEnum (D d) = fromEnum d


instance Num Distance where
    (+)
      | otherwise =
        let threshold =
                let limit = maxBound
                    width = finiteBitSize limit
                    space = 1 `shiftL` (width - 1)
                in  limit - space
        in  \case
                lhs@(D x) | x == maxBound → const lhs
                D x → \case
                    rhs@(D y) | y == maxBound → rhs
                    D y
                        | x >= threshold && y >= threshold →
                            let x' = fromIntegral x ∷ Word32
                                y' = fromIntegral x ∷ Word32
                            in  D . fromIntegral $ x' + y' `mod` 0xFFFF
                    D y →
                        let added = x + y
                        in  if added == 0xFFFF || added < x
                                then D $ added + 1
                                else D added


    (*) =
        let threshold =
                let limit = maxBound
                    width = finiteBitSize limit
                    extra = width `div` 2
                in  limit `shiftR` extra
        in  \case
                lhs@(D x) | x == maxBound → const lhs
                D x → \case
                    rhs@(D y) | y == maxBound → rhs
                    D y
                        | x >= threshold && y >= threshold →
                            let x' = fromIntegral x ∷ Word32
                                y' = fromIntegral x ∷ Word32
                            in  D . fromIntegral $ x' * y' `mod` 0xFFFF
                    D y → D $ x * y


    (-) = \case
        lhs@(D x) | x == maxBound → const lhs
        D x → \case
            D y | y == maxBound → D 0
            D y →
                let subbed = x - y
                in  if subbed > x
                        then D $ subbed - 1
                        else D subbed


    abs = id


    signum (D 0) = D 0
    signum (D _) = D 1


    negate = id


    fromInteger i = case i `compare` toInteger (maxBound ∷ Word16) of
        LT → D $ fromInteger i
        _ → maxBound


instance Read Distance where
    readPrec =
        let inf =
                get >>= \case
                    '∞' → pure infinity
                    _ → pfail
            num = D <$> readPrec
        in  inf <++ num


instance Show Distance where
    show (D d)
        | d == maxBound = "∞"
        | otherwise = show d


-- | The 'zip3' function takes three streams and returns a stream of corresponding triples.
zip3 :: NonEmpty a -> NonEmpty b -> NonEmpty c -> NonEmpty (a,b,c)
zip3 ~(x :| xs) ~(y :| ys) ~(z :| zs) = (x,y,z) :| L.zip3 xs ys zs
