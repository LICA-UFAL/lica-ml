(ns knn.core-test
  (:require [clojure.test :refer :all]
            [knn.similarity-functions :refer :all]
            [knn.core :refer :all]))


(def classifier {:k 3 :similarity-function euclidean-distance})
(def x-data [[1 1] [3 1] [1 2] [1 2]])
(def y-data [0 0 0 1])
(def sample [1 2])
(def data-features (get-features (transform-entry x-data y-data)))
(def closer-instances-test (closer-instances
      classifier sample (transform-entry x-data y-data)))

(deftest a-test
  (testing
    (is (= [
      [[1 1] 0]
      [[3 1] 0]
      [[1 2] 0]
      [[1 2] 1]
      ] (transform-entry x-data y-data))))
  (testing
    (not (empty? data-features)))
  (testing
    (not (empty? closer-instances-test)))
  (testing
    (is (= 0 (predict classifier x-data y-data sample))))
  (testing
    (is (= 1 (most-common-class [1 1 0 1 1])))))
