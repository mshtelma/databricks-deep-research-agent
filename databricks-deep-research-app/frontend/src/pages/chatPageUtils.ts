export function shouldFetchChatFullForChat(
  chatId: string | undefined,
  isDraft: boolean,
  chatExistsInApi: boolean,
): boolean {
  return !!chatId && (!isDraft || chatExistsInApi);
}
