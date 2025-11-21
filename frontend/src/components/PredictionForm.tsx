import { useState } from 'react';
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Input,
  Textarea,
  VStack,
  useToast,
  Text,
  Heading,
  Alert,
  AlertIcon,
  Progress,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
} from '@chakra-ui/react';
import { predictMutation } from '../lib/api';

interface PredictionResult {
  predicted_class: number;
  class_probabilities: {
    [key: string]: number;
  };
  gene: string;
  variation: string;
}

export default function PredictionForm() {
  const [gene, setGene] = useState('');
  const [variation, setVariation] = useState('');
  const [text, setText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const toast = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      const prediction = await predictMutation(gene, variation, text);
      setResult(prediction);
    } catch (error) {
      toast({
        title: 'Error',
        description: error instanceof Error ? error.message : 'Failed to get prediction',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box maxW="800px" mx="auto" p={6}>
      <Heading mb={6}>Personalized Medicine Classifier</Heading>
      
      <form onSubmit={handleSubmit}>
        <VStack spacing={4} align="stretch">
          <FormControl isRequired>
            <FormLabel>Gene</FormLabel>
            <Input
              value={gene}
              onChange={(e) => setGene(e.target.value)}
              placeholder="e.g., BRCA1"
            />
          </FormControl>

          <FormControl isRequired>
            <FormLabel>Variation</FormLabel>
            <Input
              value={variation}
              onChange={(e) => setVariation(e.target.value)}
              placeholder="e.g., V600E"
            />
          </FormControl>

          <FormControl isRequired>
            <FormLabel>Clinical Evidence Text</FormLabel>
            <Textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter the clinical text about the mutation..."
              rows={6}
            />
          </FormControl>

          <Button
            type="submit"
            colorScheme="blue"
            isLoading={isLoading}
            loadingText="Predicting..."
          >
            Predict
          </Button>
        </VStack>
      </form>

      {isLoading && (
        <Box mt={4}>
          <Progress size="xs" isIndeterminate />
        </Box>
      )}

      {result && (
        <Box mt={6} p={4} borderWidth={1} borderRadius="md">
          <Heading size="md" mb={4}>
            Prediction Results
          </Heading>
          
          <Alert status="info" mb={4}>
            <AlertIcon />
            <Text>
              Predicted Class: <strong>{result.predicted_class}</strong>
            </Text>
          </Alert>

          <Table variant="simple" size="sm">
            <Thead>
              <Tr>
                <Th>Class</Th>
                <Th isNumeric>Probability</Th>
              </Tr>
            </Thead>
            <Tbody>
              {Object.entries(result.class_probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([className, prob]) => (
                  <Tr key={className}>
                    <Td>{className}</Td>
                    <Td isNumeric>{(prob * 100).toFixed(2)}%</Td>
                  </Tr>
                ))}
            </Tbody>
          </Table>
        </Box>
      )}
    </Box>
  );
}