
import { quizQuestions, Axis } from '@/data/personalityQuiz';

// 回答の型: 質問IDと6段階評価の値 (1-6)
export interface Answer {
  questionId: number;
  value: number; 
}

/**
 * 回答の配列を受け取り、各軸のスコアを計算し、最終的な性格タイプを返す
 * @param answers ユーザーの回答の配列
 * @returns 16タイプ分類のキー (例: 'SUPH')
 */
export const calculatePersonalityType = (answers: Answer[]): string => {
  const scores: { [key in Axis]: number } = {
    sm: 0,
    daily: 0,
    physical: 0,
    frequency: 0,
  };

  answers.forEach(answer => {
    const question = quizQuestions.find(q => q.id === answer.questionId);
    if (question) {
      // 6段階評価(1-6)をスコア(-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)に変換して計算
      const scoreValue = answer.value - 3.5;
      scores[question.axis] += scoreValue * question.direction;
    }
  });

  // 各軸のスコアの正負でタイプを決定
  const smType = scores.sm >= 0 ? 'S' : 'M';
  const dailyType = scores.daily >= 0 ? 'U' : 'D'; // Unusual / Daily
  const physicalType = scores.physical >= 0 ? 'P' : 'M'; // Physical / Mental
  const frequencyType = scores.frequency >= 0 ? 'H' : 'L'; // High / Low

  // 4文字のタイプキーを生成
  const typeKey = `${smType}${dailyType}${physicalType}${frequencyType}`;
  console.log('Calculated scores:', scores);
  console.log('Generated type key:', typeKey);

  return typeKey;
};
