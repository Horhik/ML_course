import GHC.Float
import Data.List


type Predictor = (Int, Float)
type Vector = [Float]
type Matrix = [Vector]

data DecisionTree func = Leaf func | Node func (DecisionTree func) (DecisionTree func)

buildTree :: [Float] -> Float -> DecisionTree Predictor
