(ns knn.similarity-functions)

(defn euclidean-distance
  [first-inst second-inst]
  (* 0.5 (reduce + (mapv #(Math/pow % 2) (mapv - first-inst second-inst)))))
