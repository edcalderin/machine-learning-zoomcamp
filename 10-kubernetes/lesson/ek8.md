```bash
docker build -t model:xception-v4-001 -f image-model.dockerfile .
kind load docker-image model:xception-v4-001

kubectl apply -f kube-config/model-deployment.yaml
kubectl get pod
kubectl port-forward tf-serving-clothing-model-c8dbc8489-7vkn6 8500:8500
python predict_api.py

kubectl apply -f kube-config/model-service.yaml
kubectl get svc
kubectl port-forward service/tf-serving-clothing-model 8500:8500

docker build -t gateway:001 -f image-gateway.dockerfile .
kind load docker-image gateway:001

kubectl apply -f kube-config/gateway-deployment.yaml
kubectl get pod
kubectl port-forward gateway-77bffb87d6-jk7pq 8080:8080
python test_api.py

kubectl apply -f kube-config/gateway-service.yaml
kubectl get svc
kubectl port-forward service/gateway 9696:80
python test_api.py

ACCOUNT_ID=345098143492
AWS_REGION=us-east-2
REGISTRY_NAME=fashion-images
PREFIX=${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REGISTRY_NAME}

GATEWAY_LOCAL_IMAGE=gateway:001
GATEWAY_REMOTE_IMAGE=${PREFIX}:gateway-001
docker tag $GATEWAY_LOCAL_IMAGE $GATEWAY_REMOTE_IMAGE

MODEL_LOCAL_IMAGE=model:xception-v4-001
MODEL_REMOTE_IMAGE=${PREFIX}:model-xception-v4-001
docker tag $MODEL_LOCAL_IMAGE $MODEL_REMOTE_IMAGE

aws ecr get-login-password --region us-east-1

docker push $MODEL_REMOTE_IMAGE
docker push $GATEWAY_REMOTE_IMAGE

eksctl create cluster -f kube-config/eks-config.yaml
```
Then kubectl apply gateway/mode command

Clear resources with:

```kubectl delete cluster --name fashion-cluster```