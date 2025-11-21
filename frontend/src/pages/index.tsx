import { useState } from 'react';
import {
  Box,
  Container,
  Heading,
  VStack,
  FormControl,
  FormLabel,
  Input,  
  Textarea,
  Button,
  Text,
  useToast,
  Alert,
  AlertIcon,
  ChakraProvider,
  Progress,
  Card,
  CardBody,
} from '@chakra-ui/react';
import Head from 'next/head';

interface PredictionResult {
  predicted_class: number;
  class_probabilities: {
    [key: string]: number;
  };
  gene: string;
  variation: string;
}

const classMapping: { [key: number]: string } = {
  1: "Likely Loss-of-Function",
  2: "Likely Gain-of-Function",
  3: "Likely Neutral",
  4: "Likely Oncogenic",
  5: "Likely Benign",
  6: "Uncertain Significance",
  7: "Pathogenic",
  8: "Likely Pathogenic",
  9: "Benign"
};

export default function Home() {
  const [formData, setFormData] = useState({
    gene: '',
    variation: '',
    text: '',
  });
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const toast = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setPrediction(null);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const result: PredictionResult = await response.json();
      setPrediction(result);
      
      // Get the class with highest probability
      const classMapping = {
        1: "Likely Loss-of-Function",
        2: "Likely Gain-of-Function",
        3: "Likely Neutral",
        4: "Likely Oncogenic",
        5: "Likely Benign",
        6: "Uncertain Significance",
        7: "Pathogenic",
        8: "Likely Pathogenic",
        9: "Benign"
      };
      
      const predictedClassName = classMapping[result.predicted_class];
      
      toast({
        title: 'Prediction completed',
        description: `Predicted classification: ${predictedClassName}`,
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Error:', error);
      toast({
        title: 'An error occurred.',
        description: 'Unable to get prediction.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxW="container.md" py={10}>
      <VStack spacing={8}>
        <Heading>Personalized Medicine Model</Heading>
        
        <Box as="form" width="100%" onSubmit={handleSubmit}>
          <VStack spacing={4}>
              <FormControl id="gene" isRequired>
                <FormLabel>Gene</FormLabel>
                <Input
                placeholder="Enter gene information..."
                value={formData.gene}
                onChange={(e) => setFormData({ ...formData, gene: e.target.value })}
                size="sm"
                />
              </FormControl>

              <FormControl id="variation" isRequired>
                <FormLabel>Variation</FormLabel>
                <Input
                placeholder="Enter variation information..."
                value={formData.variation}
                onChange={(e) => setFormData({ ...formData, variation: e.target.value })}
                size="sm"
                />
              </FormControl>

              <FormControl id="text" isRequired>
                <FormLabel>Clinical Evidence</FormLabel>
                <Textarea
                placeholder="Enter clinical evidence..."
                value={formData.text}
                onChange={(e) => setFormData({ ...formData, text: e.target.value })}
                size="sm"
                />
              </FormControl>            <Button
              type="submit"
              colorScheme="blue"
              isLoading={loading}
              loadingText="Predicting..."
              width="100%"
            >
              Get Prediction
            </Button>
          </VStack>
        </Box>

        {prediction && (
          <Card width="100%">
            <CardBody>
              <VStack align="start" spacing={4}>
                <Text fontSize="xl" fontWeight="bold">Prediction Results</Text>
                <Box width="100%">
                  <Text fontWeight="semibold">Prediction: {classMapping[prediction.predicted_class as keyof typeof classMapping]}</Text>
                  <Text mt={2} fontWeight="semibold">Class Probabilities:</Text>
                  {Object.entries(prediction.class_probabilities).map(([className, prob]) => (
                    <Box key={className} mt={1}>
                      <Text>
                        {classMapping[parseInt(className.split('_')[1]) as keyof typeof classMapping]}: {(prob * 100).toFixed(1)}%
                      </Text>
                      <Progress value={prob * 100} size="sm" colorScheme={prob > 0.5 ? "green" : "blue"} />
                    </Box>
                  ))}
                </Box>
                <Text>{JSON.stringify(prediction, null, 2)}</Text>
              </VStack>
            </CardBody>
          </Card>
        )}
      </VStack>
    </Container>
  );
}