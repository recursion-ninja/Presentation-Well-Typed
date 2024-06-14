module Main (main) where

import StringAlignment

main :: IO ()
main =
    let s1 = "GATTACA"
        s2 = "ATTAGAGACA"
        result = needlemanWunschDefinition measure s1 s2
    in  putStr "Alignment distance: " *> print result
