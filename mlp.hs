import Numeric.LinearAlgebra
import System.Random
import Debug.Trace

type Mlp = (Matrix R, Matrix R, Matrix R, Matrix R)

e :: Floating a => a
e = exp 1

zeros :: Int -> Int -> Matrix R
zeros r c = (r><c) $ repeat 0

ones :: Int -> Int -> Matrix R
ones r c = (r><c) $ repeat 1

sigmoid :: Floating a => a -> a
sigmoid x = 1/(1 + e**(-x))

sigmoid' :: Matrix R -> Matrix R
sigmoid' = cmap sigmoid

randMatrix :: Int -> Int -> Double -> Double -> IO (Matrix R)
randMatrix rows cols min max = do
    gen <- newStdGen
    let (seed, _) = random gen :: (Int, StdGen)
    return $ uniformSample seed rows (replicate cols (min, max))

makeNetwork :: Matrix R -> Matrix R -> Int -> IO Mlp
makeNetwork inputs targets nhidden = do
    theta1 <- randMatrix (nin+1) nhidden min max
    theta2 <- randMatrix (nhidden+1) nout min' max'
    return (theta1, theta2, inputs', targets)
    where
        nout = cols targets
        nin = cols inputs
        min = -(1/sqrt (fromIntegral nin))
        max = 1/sqrt (fromIntegral nin)
        min' = -(1/sqrt (fromIntegral nhidden+1))
        max' = 1/sqrt (fromIntegral nhidden+1)
        inputs' = inputs ||| -ones (rows inputs) 1


forward :: Mlp -> (Matrix R, Matrix R)
forward mlp@(theta1, theta2, inputs, targets) = 
    (activations', outputs)
    where 
         activations = sigmoid' $ inputs <> theta1
         activations' = activations ||| -ones (rows activations) 1
         outputs =  sigmoid' $ activations' <> theta2

train :: Mlp -> Matrix R -> Int -> Mlp
train mlp _ 0 = mlp
train mlp@(theta1, theta2, inputs, targets) eta n =
    train (theta1', theta2', inputs, targets) eta (n-1)
    where 
          (activations, outputs) = forward mlp
          deltaO = (outputs - targets)*outputs*( 1 - outputs) -- logistic regr.
          deltaH = activations*(1 - activations)*(deltaO <> tr theta2)
          theta1' = theta1 - eta*(tr inputs <> (deltaH ?? (All, DropLast 1)))
          theta2' = theta2 - eta*(tr activations <> deltaO)

-- loads Iris dataset from UCI machine learning repository and preprocesses it.
-- the set has already been manually converted into the format accepted by
-- `loadMatrix`
loadIris :: FilePath -> IO(Matrix R, Matrix R)
loadIris path = do
    m <- loadMatrix path
    let inputs = m ?? (All, DropLast 1) 
    let classes = m ?? (All, PosCyc (idxs [-1]))
    let targets = fromLists $ map f $ toLists classes
    return (normalise inputs, targets)
    where
        f [x] = case x of
                   0 -> [1,0,0]
                   1 -> [0,1,0]
                   2 -> [0,0,1]

-- normalisation: x = (x - mean) / variance
normalise :: Matrix R -> Matrix R
normalise m =
    (m - mean') / var
    where
        (mean, covar) = meanCov m
        mean' = repmat (asRow mean) (rows m) 1
        -- the leading diagonal of the covariance matrix corresponds to
        -- each columns variance
        var = repmat (asRow $ takeDiag covar) (rows m) 1

-- transforms each output vector into a `1 of N` output
trigger :: Vector R -> Vector R
trigger vect =
    assoc (size vect) 0 [(maxIndex vect,1)]

-- computes correcteness percentage 
correcteness :: Mlp -> Double
correcteness mlp@(_, _, inputs, targets) =
    100 * (sumElements (sums*targets) / fromIntegral (rows inputs))
    where
        (_, predictions) = forward mlp
        sums = fromRows $ map trigger 
            $ toRows predictions

main :: IO ()
main = do
    --let inputs = matrix 1 $ toList (linspace 20 (1,5)) -- REGRESSION EXAMPLE
    --let targets = matrix 1 [sin x | x <- toList $ flatten inputs] 

    (inputs, targets) <- loadIris "iris.data"
    mlp <- makeNetwork inputs targets 5
    let eta = scalar 0.25
    let trained = train mlp eta 100
    let perc = correcteness trained
    print $ "correct: " ++ show perc ++ "%"
